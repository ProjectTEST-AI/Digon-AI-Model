"""
Copyright (c) 2025 The ExpG3 Authors
All Rights Reserved.

PROPRIETARY AND CONFIDENTIAL

This software is the proprietary and confidential property of the copyright holder.
Possession and use of this software is strictly limited by the terms of a separate license agreement.

UNAUTHORIZED COPYING, DISTRIBUTION, OR USE OF THIS SOFTWARE, OR ANY PORTION OF IT, IS STRICTLY PROHIBITED.

This software is provided "as-is" without warranty of any kind, express or implied.
"""
# -*- coding: utf-8 -*-
"""
Experimental G3 Model Training Script 
Refactored for cleaner code, better variable naming, and consistent Keras Model naming.
TPU-adapted, streaming FineWeb-Edu from Hugging Face Datasets, with text generation.
EOS token used for padding. Architectural change for memory gradient flow.
Memory-saving hyperparameters enabled for TPU testing.
Removed semicolons.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
import tqdm
from math import prod
import keras # Keras 3
keras.config.disable_traceback_filtering() 
from tensorflow.keras import mixed_precision # Using tf.keras for mixed_precision
import time
import traceback
import os
import json

# Hugging Face Libraries
from datasets import load_dataset 
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from transformers import PreTrainedTokenizerFast

# --- Global flag to track TPU initialization ---
_TPU_INITIALIZED_IN_SESSION = False
_LAST_TPU_RESOLVER = None

def cleanup_tensorflow_runtimes():
    """Attempts to clean up TensorFlow and Keras states for re-running in same session."""
    global _TPU_INITIALIZED_IN_SESSION, _LAST_TPU_RESOLVER
    print("Attempting to clean up TensorFlow and Keras runtime state...")
    try:
        keras.backend.clear_session()
        print("Keras session cleared.")
    except Exception as e:
        print(f"Error clearing Keras session: {e}")
    try:
        tf.compat.v1.reset_default_graph()
        print("TensorFlow default graph reset.")
    except Exception as e:
        print(f"Error resetting default graph: {e}")
    
    if _TPU_INITIALIZED_IN_SESSION and _LAST_TPU_RESOLVER is not None:
        resolver_master_info = _LAST_TPU_RESOLVER.master() if _LAST_TPU_RESOLVER else 'N/A'
        print(f"Note: Previous TPU resolver was {resolver_master_info}. "
              "True TPU system shutdown from Python is complex and often not fully supported. "
              "A kernel restart is the most reliable way to reset TPU state.")
    
    _TPU_INITIALIZED_IN_SESSION = False
    _LAST_TPU_RESOLVER = None
    print("Cleanup attempt finished. For full reset, especially for TPUs, restart the kernel.")

# =========================================================================================
# =                                Mixed Precision Setup                                  =
# =========================================================================================
policy_name = 'mixed_bfloat16'
try:
    _ = tf.constant(1.0, dtype=tf.bfloat16)
    print(f"Using {policy_name} policy.")
except Exception:
    print(f"Warning: bfloat16 test failed. Falling back to mixed_float16.")
    policy_name = 'mixed_float16'
policy = mixed_precision.Policy(policy_name)
mixed_precision.set_global_policy(policy)
print(f"Global mixed precision policy set to: {mixed_precision.global_policy().name}")

# Importing layers directly from keras (Keras 3)
from keras.layers import ( 
    Input, LayerNormalization, Concatenate, TimeDistributed, DepthwiseConv1D, Conv2D,
    Activation, Add, Dense, Lambda, Multiply, Conv1D, Embedding, AveragePooling1D
)

# =========================================================================================
# =                            Hyperparameters (TPU Adjusted)                           =
# =========================================================================================
VOCAB_SIZE_TARGET = 32000
EMBEDDING_DIM = 2048
MAX_SEQ_LEN = 512 
NUM_REGISTERS = 512 
MEMORY_DIM = EMBEDDING_DIM

# --- Memory Saving Configuration for TPU testing ---
NUM_MODEL_LAYERS = 6 
NUM_MTP_HEADS = 0
SSM_STATE_DIM_MULTIPLIER = 1 
# --- End Memory Saving Configuration ---

SSM_RANK_DIVISOR = 4 

BATCH_SIZE_PER_REPLICA = 16 
EPOCHS = 1 
LEARNING_RATE = 1e-4
WARMUP_STEPS = 10 
CLIP_NORM = 1.0

TOKENIZER_SAVE_DIR = "./housecat_tokenizer_hf_fineweb_eos_pad"
TOKENIZER_CONFIG_FILENAME = "tokenizer.json" 

PAD_TOKEN_ID = 0 
EOS_TOKEN_ID = 1
ACTUAL_VOCAB_SIZE = VOCAB_SIZE_TARGET 

HF_DATASET_NAME = "HuggingFaceFW/fineweb-edu"
HF_DATASET_SPLIT = "train"
SAMPLES_FOR_TOKENIZER_TRAINING = 10000 
FORCE_RETRAIN_TOKENIZER = False 
TOTAL_TRAIN_STEPS_CONFIG = 1

# =========================================================================================
# =                         Tokenizer Training & Loading
# =========================================================================================
def get_text_iterator_from_hf_stream(dataset_name, split, num_samples=None, text_column="text"):
    print(f"Setting up text iterator for '{dataset_name}/{split}' for {num_samples or 'all'} samples.")
    streamed_dataset = load_dataset(dataset_name, streaming=True, split=split, trust_remote_code=True)
    count = 0
    for example in streamed_dataset:
        if text_column in example and example[text_column]:
            yield example[text_column]
            count += 1
            if num_samples is not None and count >= num_samples:
                print(f"Reached specified {num_samples} samples for iterator.")
                break
        elif num_samples is not None and count >= num_samples: 
             print(f"Reached specified {num_samples} samples (iterator might have empty items).")
             break
    if num_samples is not None and count < num_samples:
        print(f"Warning: Text iterator yielded only {count} samples, less than requested {num_samples}.")

def train_and_save_tokenizer_if_needed(
    text_iterator_fn, 
    target_vocab_size, 
    tokenizer_output_dir, 
    tokenizer_filename,
    force_retrain=False
):
    global PAD_TOKEN_ID, EOS_TOKEN_ID 
    os.makedirs(tokenizer_output_dir, exist_ok=True)
    config_file_path = os.path.join(tokenizer_output_dir, tokenizer_filename)

    if not force_retrain and os.path.exists(config_file_path):
        print(f"Tokenizer already trained. Loading from {config_file_path}")
        try:
            tokenizer_object = Tokenizer.from_file(config_file_path)
            loaded_eos_id = tokenizer_object.token_to_id("[EOS]")
            EOS_TOKEN_ID = loaded_eos_id if loaded_eos_id is not None else 2 
            PAD_TOKEN_ID = EOS_TOKEN_ID 
            print(f"Successfully loaded tokenizer. EOS ID: {EOS_TOKEN_ID}, PAD ID (set to EOS): {PAD_TOKEN_ID}")
            return tokenizer_object
        except Exception as e:
            print(f"Error loading existing tokenizer from '{config_file_path}': {e}. Will attempt to retrain.")

    print("--- Starting New Tokenizer Training ---")
    tokenizer_object = Tokenizer(models.BPE())
    tokenizer_object.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer_object.decoder = decoders.ByteLevel()
    tokenizer_object.post_processor = processors.ByteLevel(trim_offsets=True)

    special_tokens_list = ["[UNK]", "[EOS]", "[MASK]", "[PAD]"] 
    bpe_trainer = trainers.BpeTrainer(
        vocab_size=target_vocab_size,
        min_frequency=2,
        show_progress=True,
        special_tokens=special_tokens_list
    )
    print("Requesting text corpus iterator for tokenizer training...")
    text_corpus_iterator = text_iterator_fn() 
    print("Starting tokenizer.train_from_iterator()...")
    training_start_time = time.time()
    try:
        tokenizer_object.train_from_iterator(text_corpus_iterator, trainer=bpe_trainer)
    except Exception as e:
        print(f"ERROR during tokenizer training from iterator: {e}")
        print(traceback.format_exc())
        return None 
    training_end_time = time.time()
    print(f"Tokenizer training completed in {training_end_time - training_start_time:.2f} seconds.")

    trained_eos_id = tokenizer_object.token_to_id("[EOS]")
    if trained_eos_id is None:
        print("Warning: [EOS] token not found in trained tokenizer vocab! Adding it.")
        tokenizer_object.add_special_tokens(["[EOS]"]) 
        trained_eos_id = tokenizer_object.token_to_id("[EOS]")
        if trained_eos_id is None: 
            print("CRITICAL: Still cannot find [EOS] token after adding. Defaulting EOS ID to 2.")
            EOS_TOKEN_ID = 2 
        else:
            EOS_TOKEN_ID = trained_eos_id
    else:
        EOS_TOKEN_ID = trained_eos_id
    
    PAD_TOKEN_ID = EOS_TOKEN_ID 
    print(f"EOS ID: {EOS_TOKEN_ID}, PAD ID (set to EOS): {PAD_TOKEN_ID} (from newly trained tokenizer)")

    tokenizer_object.save(config_file_path)
    print(f"Tokenizer saved to {config_file_path}")
    return tokenizer_object

tokenizer = None # Global tokenizer instance, initialized by initialize_tokenizer()

def initialize_tokenizer():
    global tokenizer, EOS_TOKEN_ID, PAD_TOKEN_ID, ACTUAL_VOCAB_SIZE
    print("--- Tokenizer Initialization ---")
    tokenizer_config_path = os.path.join(TOKENIZER_SAVE_DIR, TOKENIZER_CONFIG_FILENAME)

    if SAMPLES_FOR_TOKENIZER_TRAINING > 0:
        print(f"Attempting to train tokenizer on first {SAMPLES_FOR_TOKENIZER_TRAINING} samples from HF stream.")
        hf_tokenizer_object = train_and_save_tokenizer_if_needed(
            text_iterator_fn=lambda: get_text_iterator_from_hf_stream(
                HF_DATASET_NAME, HF_DATASET_SPLIT, SAMPLES_FOR_TOKENIZER_TRAINING
            ),
            target_vocab_size=VOCAB_SIZE_TARGET,
            tokenizer_output_dir=TOKENIZER_SAVE_DIR,
            tokenizer_filename=TOKENIZER_CONFIG_FILENAME,
            force_retrain=FORCE_RETRAIN_TOKENIZER
        )
    else:
        print("Skipping tokenizer training from stream as SAMPLES_FOR_TOKENIZER_TRAINING is not positive.")
        hf_tokenizer_object = None
        if os.path.exists(tokenizer_config_path):
            try:
                hf_tokenizer_object = Tokenizer.from_file(tokenizer_config_path)
                loaded_eos_id = hf_tokenizer_object.token_to_id("[EOS]")
                EOS_TOKEN_ID = loaded_eos_id if loaded_eos_id is not None else 2
                PAD_TOKEN_ID = EOS_TOKEN_ID
                print(f"Successfully loaded pre-trained tokenizer from {tokenizer_config_path}.")
            except Exception as e:
                print(f"Failed to load pre-trained tokenizer from {tokenizer_config_path}: {e}")
                hf_tokenizer_object = None
        else:
            print(f"Pre-trained tokenizer file not found at {tokenizer_config_path}.")

    if hf_tokenizer_object is None:
        print("CRITICAL: Tokenizer object (from huggingface/tokenizers) is unavailable. Cannot proceed.")
        return 
    
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=hf_tokenizer_object,
        eos_token="[EOS]",
        unk_token="[UNK]",
        mask_token="[MASK]",
        pad_token="[EOS]" 
    )
    
    EOS_TOKEN_ID = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2
    PAD_TOKEN_ID = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else EOS_TOKEN_ID
    ACTUAL_VOCAB_SIZE = tokenizer.vocab_size

    if EOS_TOKEN_ID != PAD_TOKEN_ID:
        print(f"Warning: EOS_TOKEN_ID ({EOS_TOKEN_ID}) != PAD_TOKEN_ID ({PAD_TOKEN_ID}) after PreTrainedTokenizerFast init.")
        print("Forcing PAD_TOKEN_ID to be the same as EOS_TOKEN_ID for model consistency.")
        PAD_TOKEN_ID = EOS_TOKEN_ID
    
    print(f"Tokenizer (PreTrainedTokenizerFast) ready. Actual Vocab Size: {ACTUAL_VOCAB_SIZE}, EOS ID: {EOS_TOKEN_ID}, PAD ID: {PAD_TOKEN_ID}")

# =========================================================================================
# =                                    Keras Layers
# =========================================================================================
class FourierFeatures(keras.layers.Layer):
    def __init__(self, embedding_dimension, name="fourier_features", **kwargs):
        super().__init__(name=name, **kwargs)
        self.embedding_dimension = embedding_dimension

    def call(self, inputs_tensor):
        input_tensor_shape = tf.shape(inputs_tensor)
        batch_size = input_tensor_shape[0]
        sequence_length = input_tensor_shape[1]
        
        position_indices = tf.range(sequence_length, dtype=tf.float32)[:, None] 
        
        half_embedding_dimension = self.embedding_dimension // 2
        div_term_denominator = tf.cast(half_embedding_dimension, tf.float32) 
        div_term_values = tf.pow(10000.0, tf.range(0, half_embedding_dimension, dtype=tf.float32) / div_term_denominator) 
        
        angle_values = position_indices / div_term_values 
        
        fourier_components = tf.concat([tf.sin(angle_values), tf.cos(angle_values)], axis=-1) 
        
        if self.embedding_dimension % 2 != 0:
            fourier_components = tf.pad(fourier_components, [[0, 0], [0, 1]]) 
            
        expanded_fourier_components = tf.expand_dims(fourier_components, 0) 
        tiled_fourier_features = tf.tile(expanded_fourier_components, [batch_size, 1, 1])    
        
        return tf.cast(tiled_fourier_features, dtype=inputs_tensor.dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"embedding_dimension": self.embedding_dimension})
        return config

class FusedPositionalEncoding(keras.layers.Layer):
    def __init__(self, embedding_dimension, max_sequence_length, dynamic_rope=False, name="fused_positional_encoding", **kwargs):
        super().__init__(name=name, **kwargs)
        self.embedding_dimension = embedding_dimension
        self.max_sequence_length = max_sequence_length
        self.dynamic_rope = dynamic_rope 
        if self.embedding_dimension % 2 != 0:
            raise ValueError(f"Embedding dimension ({self.embedding_dimension}) must be even for RoPE.")

    def build(self, input_tensor_shape): 
        if not self.dynamic_rope:
            inverse_frequencies = self._compute_inverse_frequencies() 
            positions_range = tf.range(self.max_sequence_length, dtype=inverse_frequencies.dtype) 
            scaled_time_values = tf.einsum('i,j->ij', positions_range, inverse_frequencies)
            rope_embeddings_precomputed = tf.concat([tf.sin(scaled_time_values), tf.cos(scaled_time_values)], axis=-1)
            
            self.static_rope_frequencies = self.add_weight(
                name="static_rope_frequencies_weight", 
                shape=rope_embeddings_precomputed.shape,
                dtype=self.dtype_policy.variable_dtype, 
                initializer=keras.initializers.Constant(tf.cast(rope_embeddings_precomputed, self.dtype_policy.variable_dtype)),
                trainable=False
            )
        
        self.global_positional_embedding_layer = Embedding(
            input_dim=self.max_sequence_length, 
            output_dim=self.embedding_dimension, 
            name="global_positional_embedding_lookup"
        )
        if not self.global_positional_embedding_layer.built: 
             self.global_positional_embedding_layer.build(tf.TensorShape((None, self.max_sequence_length))) 
        super().build(input_tensor_shape)

    def _compute_inverse_frequencies(self):
        half_dimension = self.embedding_dimension // 2
        thermostat_base = 10000.0
        exponent_values = (tf.range(0, half_dimension, dtype=tf.float32) * 2.0) / tf.cast(self.embedding_dimension, tf.float32)
        return 1.0 / (thermostat_base ** exponent_values)

    def _apply_rotary_embeddings(self, query_tensor, rope_embeddings_for_sequence):
        query_dtype = query_tensor.dtype
        rope_embeddings_for_sequence = tf.cast(rope_embeddings_for_sequence, query_dtype)

        query_tf_shape = tf.shape(query_tensor)
        batch_size, sequence_length = query_tf_shape[0], query_tf_shape[1]
        half_dimension = self.embedding_dimension // 2

        query_reshaped_for_rotation = tf.reshape(query_tensor, [batch_size, sequence_length, half_dimension, 2])
        query_real_parts = query_reshaped_for_rotation[..., 0]
        query_imag_parts = query_reshaped_for_rotation[..., 1]

        sin_terms, cos_terms = tf.split(rope_embeddings_for_sequence, num_or_size_splits=2, axis=-1)
        
        sin_terms_broadcast = sin_terms[None, :, :] 
        cos_terms_broadcast = cos_terms[None, :, :] 

        rotated_real_parts = query_real_parts * cos_terms_broadcast - query_imag_parts * sin_terms_broadcast
        rotated_imag_parts = query_real_parts * sin_terms_broadcast + query_imag_parts * cos_terms_broadcast
        
        rotated_query_stacked = tf.stack([rotated_real_parts, rotated_imag_parts], axis=-1)
        return tf.reshape(rotated_query_stacked, query_tf_shape)

    def call(self, sequence_input_tensor):
        input_tf_shape = tf.shape(sequence_input_tensor)
        batch_size, current_sequence_len = input_tf_shape[0], input_tf_shape[1]

        if self.dynamic_rope:
            inv_freqs = self._compute_inverse_frequencies()
            pos_range = tf.range(current_sequence_len, dtype=inv_freqs.dtype)
            scaled_time_vals = tf.einsum('i,j->ij', pos_range, inv_freqs)
            rope_embeddings_current_seq = tf.concat([tf.sin(scaled_time_vals), tf.cos(scaled_time_vals)], axis=-1)
        else:
            rope_embeddings_current_seq = self.static_rope_frequencies[:current_sequence_len, :]
        
        sequence_with_rope_applied = self._apply_rotary_embeddings(sequence_input_tensor, rope_embeddings_current_seq)
        
        position_indices_for_gpe = tf.range(current_sequence_len)
        batched_indices_for_gpe = tf.tile(position_indices_for_gpe[None, :], [batch_size, 1])
        global_positional_values = self.global_positional_embedding_layer(batched_indices_for_gpe)
        
        return sequence_with_rope_applied + tf.cast(global_positional_values, sequence_input_tensor.dtype)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embedding_dimension": self.embedding_dimension,
            "max_sequence_length": self.max_sequence_length,
            "dynamic_rope": self.dynamic_rope
        })
        return config

class ConditionalAveragePooling1D(keras.layers.Layer): 
    def __init__(self, pool_size=2, strides=2, padding_if_pool='SAME', name="conditional_avg_pool", **kwargs):
        super().__init__(name=name, **kwargs)
        self.pool_size = pool_size
        self.strides = strides
        self.padding_if_pool = padding_if_pool.upper() 
        self.avg_pooling_layer = AveragePooling1D(
            pool_size=self.pool_size, 
            strides=self.strides, 
            padding=self.padding_if_pool
        )
    
    def call(self, inputs_tensor):
        current_sequence_length = tf.shape(inputs_tensor)[1]
        
        if self.padding_if_pool == 'SAME':
            condition_to_skip_pooling_op = current_sequence_length <= 1 
        else: # 'VALID'
            condition_to_skip_pooling_op = current_sequence_length < self.pool_size
            
        return tf.cond(
            condition_to_skip_pooling_op, 
            lambda: inputs_tensor,                      
            lambda: self.avg_pooling_layer(inputs_tensor) 
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "pool_size": self.pool_size, 
            "strides": self.strides, 
            "padding_if_pool": self.padding_if_pool
        })
        return config

class ExpandDimsLayer(keras.layers.Layer):
    def __init__(self, axis, name="expand_dims_layer", **kwargs):
        super().__init__(name=name, **kwargs)
        self.axis_to_expand = axis
    def call(self, inputs_tensor): return tf.expand_dims(inputs_tensor, self.axis_to_expand)
    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis_to_expand})
        return config

class SqueezeLayer(keras.layers.Layer):
    def __init__(self, axis=None, name="squeeze_layer", **kwargs):
        super().__init__(name=name, **kwargs)
        self.axis_to_squeeze = axis
    def call(self, inputs_tensor): return tf.squeeze(inputs_tensor, self.axis_to_squeeze)
    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis_to_squeeze})
        return config

class CausalPadding1D(keras.layers.Layer):
    def __init__(self, kernel_size, dilation_rate=1, name="causal_padding_1d", **kwargs):
        super().__init__(name=name, **kwargs)
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.left_padding_amount = dilation_rate * (kernel_size - 1)
    def call(self, inputs_tensor):
        return tf.pad(inputs_tensor,[[0,0],[self.left_padding_amount,0],[0,0]])
    def get_config(self):
        config = super().get_config()
        config.update({"kernel_size": self.kernel_size, "dilation_rate": self.dilation_rate})
        return config

class CausalPadding2D(keras.layers.Layer): 
    def __init__(self, kernel_size_tuple, name="causal_padding_2d", **kwargs): 
        super().__init__(name=name, **kwargs)
        self.time_kernel_size = kernel_size_tuple[0]
        self.feature_kernel_size = kernel_size_tuple[1]
        self.time_padding_amount = self.time_kernel_size - 1
        feature_padding_total = self.feature_kernel_size - 1 
        self.feature_pad_before = feature_padding_total // 2
        self.feature_pad_after = feature_padding_total - self.feature_pad_before
    def call(self, inputs_tensor):
        return tf.pad(inputs_tensor, [
            [0,0], 
            [self.time_padding_amount,0], 
            [self.feature_pad_before,self.feature_pad_after], 
            [0,0]  
        ])
    def get_config(self):
        config = super().get_config()
        config.update({"kernel_size_tuple":(self.time_kernel_size, self.feature_kernel_size)})
        return config

class AutoNWayEmbedding(keras.layers.Layer): 
    def __init__(self, vocab_size, embedding_dimension, num_factors, name="autonway_embedding", **kwargs):
        super().__init__(name=name, **kwargs)
        self.vocab_size = vocab_size
        self.embedding_dimension = embedding_dimension
        self.num_factors = num_factors
        if self.embedding_dimension < 0 or self.num_factors < 0: raise ValueError("Emb_dim/factors non-negative.")
        if self.embedding_dimension > 0 and self.num_factors == 0: raise ValueError("Positive emb_dim with zero num_factors.")
        
        if self.embedding_dimension == 0: self.factor_dimensions = tuple([0] * num_factors) if self.num_factors > 0 else tuple()
        else: self.factor_dimensions = self._find_optimal_dimensions(self.embedding_dimension, self.num_factors)
        
        current_product = prod(d for d in self.factor_dimensions if d > 0) 
        target_product = self.embedding_dimension if self.embedding_dimension > 0 else 1
        
        if current_product != target_product:
            if self.num_factors > 0 and self.embedding_dimension > 0:
                base_dims = [1] * self.num_factors
                base_dims[-1] = self.embedding_dimension
                for i in range(self.num_factors - 1): 
                    root_val = int(round(base_dims[-1]**(1.0 / (self.num_factors - i))))
                    if root_val > 1 and base_dims[-1] % root_val == 0: 
                        base_dims[i] = root_val
                        base_dims[-1] //= root_val
                self.factor_dimensions = tuple(base_dims)
            current_product = prod(d for d in self.factor_dimensions if d > 0)
            if current_product != target_product: 
                raise ValueError(f"Factorization fallback fail. Target:{self.embedding_dimension},NFactors:{self.num_factors},Got:{self.factor_dimensions},Prod:{current_product}")
        
        self.embedding_factors_list = []
        for i, factor_dim_value in enumerate(self.factor_dimensions):
            if factor_dim_value > 0:
                self.embedding_factors_list.append(
                    Embedding(self.vocab_size, factor_dim_value, name=f"{self.name}_factor_{i+1}")
                )

    def _find_optimal_dimensions(self, target_product_val, num_factors_to_find_val):
        if num_factors_to_find_val <= 0: return tuple()
        if target_product_val == 0: return tuple([0] * num_factors_to_find_val)
        if num_factors_to_find_val == 1: return (target_product_val,)
        
        dimensions_list = [1] * num_factors_to_find_val
        prime_factors_map = {}
        divisor = 2
        n_temp = target_product_val
        while divisor * divisor <= n_temp:
            while n_temp % divisor == 0:
                prime_factors_map[divisor] = prime_factors_map.get(divisor, 0) + 1
                n_temp //= divisor
            divisor += 1
        if n_temp > 1:
            prime_factors_map[n_temp] = prime_factors_map.get(n_temp, 0) + 1
        
        for prime_factor_key in sorted(prime_factors_map.keys(), reverse=True):
            for _ in range(prime_factors_map[prime_factor_key]):
                dimensions_list.sort()
                dimensions_list[0] *= prime_factor_key
        dimensions_list.sort(reverse=True)
        return tuple(dimensions_list)

    def build(self, input_shape_for_ids): 
        for emb_layer_instance in self.embedding_factors_list:
            if not emb_layer_instance.built: 
                emb_layer_instance.build(input_shape_for_ids)
        super().build(input_shape_for_ids)

    def call(self, token_ids_tensor_input):
        token_ids_as_int32 = tf.cast(token_ids_tensor_input, tf.int32)
        if not self.embedding_factors_list or self.embedding_dimension == 0:
            shape_of_token_ids = tf.shape(token_ids_tensor_input)
            return tf.zeros((shape_of_token_ids[0], shape_of_token_ids[1], self.embedding_dimension), dtype=self.compute_dtype)
        
        list_of_factor_embeddings = [layer_instance(token_ids_as_int32) for layer_instance in self.embedding_factors_list]
        if not list_of_factor_embeddings: 
            shape_of_token_ids = tf.shape(token_ids_tensor_input)
            return tf.zeros((shape_of_token_ids[0], shape_of_token_ids[1], self.embedding_dimension), dtype=self.compute_dtype)
        
        current_combined_embedding = list_of_factor_embeddings[0]
        list_of_valid_factor_dims = [dim_val for dim_val in self.factor_dimensions if dim_val > 0]
        product_of_dims_so_far = list_of_valid_factor_dims[0]

        for i in range(1, len(list_of_factor_embeddings)):
            next_factor_embedding_tensor = list_of_factor_embeddings[i]
            expanded_current_combined = tf.expand_dims(current_combined_embedding, axis=-1)
            expanded_next_factor = tf.expand_dims(next_factor_embedding_tensor, axis=-2)
            current_combined_embedding = expanded_current_combined * expanded_next_factor
            
            dimension_of_next_factor = list_of_valid_factor_dims[i]
            new_total_product_dimension = product_of_dims_so_far * dimension_of_next_factor
            
            shape_of_original_ids = tf.shape(token_ids_as_int32)
            current_combined_embedding = tf.reshape(
                current_combined_embedding, 
                (shape_of_original_ids[0], shape_of_original_ids[1], new_total_product_dimension)
            )
            product_of_dims_so_far = new_total_product_dimension
            
        return tf.cast(current_combined_embedding, self.compute_dtype)

    def compute_logits(self, hidden_states_input_tensor):
        if not self.embedding_factors_list or self.embedding_dimension == 0:
            return Lambda(
                lambda h_states: tf.zeros((tf.shape(h_states)[0], tf.shape(h_states)[1], self.vocab_size), dtype=self.compute_dtype), 
                name=f"{self.name}_zeros_logits"
            )(hidden_states_input_tensor)

        list_of_factor_weight_matrices = [tf.cast(layer.embeddings, self.compute_dtype) for layer in self.embedding_factors_list]
        if not list_of_factor_weight_matrices:
            return Lambda(
                lambda h_states: tf.zeros((tf.shape(h_states)[0], tf.shape(h_states)[1], self.vocab_size), dtype=self.compute_dtype), 
                name=f"{self.name}_zeros_logits_fallback"
            )(hidden_states_input_tensor)

        current_combined_weights_matrix = list_of_factor_weight_matrices[0] 
        list_of_valid_factor_dims = [dim_val for dim_val in self.factor_dimensions if dim_val > 0]
        product_of_dims_so_far = list_of_valid_factor_dims[0]

        for i in range(1, len(list_of_factor_weight_matrices)):
            next_factor_matrix = list_of_factor_weight_matrices[i]
            expanded_current_combined_weights = tf.expand_dims(current_combined_weights_matrix, axis=-1) 
            expanded_next_factor_weights = tf.expand_dims(next_factor_matrix, axis=-2) 
            current_combined_weights_matrix = expanded_current_combined_weights * expanded_next_factor_weights
            
            dimension_of_next_factor = list_of_valid_factor_dims[i]
            new_total_product_dimension = product_of_dims_so_far * dimension_of_next_factor
            current_combined_weights_matrix = tf.reshape(
                current_combined_weights_matrix,
                (self.vocab_size, new_total_product_dimension) 
            )
            product_of_dims_so_far = new_total_product_dimension
        
        return Lambda(
            lambda hidden_s: tf.matmul(hidden_s, current_combined_weights_matrix, transpose_b=True), 
            name=f"{self.name}_matmul_logits"
        )(hidden_states_input_tensor)

    def get_config(self): 
        config = super().get_config()
        config.update({
            "vocab_size":self.vocab_size, 
            "embedding_dimension":self.embedding_dimension, 
            "num_factors":self.num_factors
        })
        return config

class StatefulMemoryModule(keras.layers.Layer): 
    def __init__(self, memory_size_slots, memory_slot_dimension, name="stateful_memory_module", **kwargs):
        super().__init__(name=name, **kwargs)
        self.memory_size_slots = memory_size_slots
        self.memory_slot_dimension = memory_slot_dimension
        
        self.update_gate_dense_layer = Dense(memory_slot_dimension, activation='sigmoid', name=f"{self.name}_update_gate_dense")
        self.reset_gate_dense_layer = Dense(memory_slot_dimension, activation='sigmoid', name=f"{self.name}_reset_gate_dense")
        self.candidate_state_conv1d_layer = Conv1D(
            filters=memory_slot_dimension, 
            kernel_size=1, 
            padding='same', # To maintain memory_size_slots dimension
            activation='tanh',
            name=f"{self.name}_candidate_state_conv1d"
        )

    def build(self, input_shapes_tuple): # Expects ((query_shape), (memory_state_shape))
        # Shape for layers operating on each memory slot independently: (batch, memory_size_slots, memory_slot_dimension)
        cell_input_output_tensor_shape = tf.TensorShape((None, self.memory_size_slots, self.memory_slot_dimension))
        
        if not self.update_gate_dense_layer.built: self.update_gate_dense_layer.build(cell_input_output_tensor_shape)
        if not self.reset_gate_dense_layer.built: self.reset_gate_dense_layer.build(cell_input_output_tensor_shape)
        if not self.candidate_state_conv1d_layer.built: self.candidate_state_conv1d_layer.build(cell_input_output_tensor_shape)
        super().build(input_shapes_tuple)

    def _gru_cell_like_update_logic(self, tiled_query_tensor_per_slot, previous_memory_state_tensor_per_slot):
        gate_input_tensor = tiled_query_tensor_per_slot + previous_memory_state_tensor_per_slot
        
        update_gate_activations = self.update_gate_dense_layer(gate_input_tensor)
        reset_gate_activations = self.reset_gate_dense_layer(gate_input_tensor)
        
        candidate_generation_input_tensor = tiled_query_tensor_per_slot + (reset_gate_activations * previous_memory_state_tensor_per_slot)
        candidate_memory_slot_values_tensor = self.candidate_state_conv1d_layer(candidate_generation_input_tensor)
        
        new_memory_state_tensor = (1.0 - update_gate_activations) * previous_memory_state_tensor_per_slot + \
                                  update_gate_activations * candidate_memory_slot_values_tensor
        return new_memory_state_tensor

    def read_from_memory_slots(self, query_input_tensor, current_memory_state_tensor):
        attention_scores_unscaled_values = tf.matmul(query_input_tensor, current_memory_state_tensor, transpose_b=True)
        
        scaling_value = tf.sqrt(tf.cast(self.memory_slot_dimension, query_input_tensor.dtype))
        attention_scores_scaled_values = attention_scores_unscaled_values / tf.maximum(scaling_value, keras.backend.epsilon())
        
        attention_distribution_weights_tensor = tf.nn.softmax(attention_scores_scaled_values, axis=-1)
        
        retrieved_information_vector = tf.matmul(attention_distribution_weights_tensor, current_memory_state_tensor)
        return retrieved_information_vector

    def write_to_memory_slots(self, single_query_input_tensor, previous_memory_state_tensor):
        # Tile single query (e.g., last sequence token) to all memory slots
        tiling_multiples = tf.stack([1, self.memory_size_slots, 1])
        tiled_query_for_all_slots = tf.tile(single_query_input_tensor, tiling_multiples)
        
        updated_memory_state_tensor = self._gru_cell_like_update_logic(tiled_query_for_all_slots, previous_memory_state_tensor)
        return updated_memory_state_tensor

    def call(self, inputs_tuple, mode='write', training=None): # Add training arg for Keras convention
        query_tensor, memory_state_tensor = inputs_tuple
        
        query_tensor_casted = tf.cast(query_tensor, self.compute_dtype)
        memory_state_tensor_casted = tf.cast(memory_state_tensor, self.compute_dtype)

        if mode == 'read':
            output_vector_from_read_op = self.read_from_memory_slots(query_tensor_casted, memory_state_tensor_casted)
            output_memory_state_op = memory_state_tensor_casted # Read does not modify memory
        elif mode == 'write':
            output_memory_state_op = self.write_to_memory_slots(query_tensor_casted, memory_state_tensor_casted)
            # For sequence processing, write op often returns a placeholder (like zeros) for the sequence path
            output_vector_from_read_op = tf.zeros_like(query_tensor_casted, dtype=self.compute_dtype) 
        else:
            raise ValueError(f"Invalid mode for {self.name}: '{mode}'. Must be 'read' or 'write'.")
        return output_vector_from_read_op, output_memory_state_op

    def get_config(self):
        config = super().get_config()
        config.update({
            "memory_size_slots": self.memory_size_slots,
            "memory_slot_dimension": self.memory_slot_dimension
        })
        return config

class SSMCore(keras.layers.Layer): 
    def __init__(self, model_dimension, state_dimension_config, ssm_rank_config, name="ssm_core", **kwargs): # Corrected ssm_rank_config usage
        super().__init__(name=name, **kwargs)
        self.model_dimension = model_dimension
        self.s4_rank = ssm_rank_config # CORRECTED: ssm_rank_config IS the rank value
        self.configured_state_dimension = state_dimension_config # Store user-provided value for get_config

        assert state_dimension_config >= 0, "State dimension must be non-negative."
        # Internal state dimension 'N', rounded up to power of 2 if > 0
        if state_dimension_config > 0 and (state_dimension_config & (state_dimension_config - 1) != 0):
            self.internal_state_dimension_N = 1 << (state_dimension_config - 1).bit_length()
        else:
            self.internal_state_dimension_N = state_dimension_config
            
        self.num_butterfly_levels = int(np.log2(self.internal_state_dimension_N)) if self.internal_state_dimension_N > 0 else 0
        self.A_butterfly_factors_list = [] # To store trainable (2,2) matrices for A

    def build(self, input_tensor_shape_ignored): # input_shape not strictly needed for deferred building of sub-layers if shapes are fixed
        variable_dtype_for_weights = self.dtype_policy.variable_dtype
        
        if self.internal_state_dimension_N > 0:
            for i in range(self.num_butterfly_levels):
                factor_matrix_weight = self.add_weight(
                    name=f"{self.name}_A_butterfly_factor_{i}",
                    shape=(2, 2), initializer="orthogonal", trainable=True, dtype=variable_dtype_for_weights
                )
                self.A_butterfly_factors_list.append(factor_matrix_weight)
        
        self.B_project_to_rank_dense = Dense(self.s4_rank, use_bias=False, name=f"{self.name}_B_project_to_rank")
        self.B_expand_to_state_dense = Dense(self.internal_state_dimension_N, use_bias=False, name=f"{self.name}_B_expand_to_state")
        
        self.C_project_to_rank_dense = Dense(self.s4_rank, use_bias=False, name=f"{self.name}_C_project_to_rank") if self.internal_state_dimension_N > 0 else None
        self.C_expand_to_model_dense = Dense(self.model_dimension, use_bias=False, name=f"{self.name}_C_expand_to_model")
        
        self.D_feedthrough_vector_weight = self.add_weight(
            name=f"{self.name}_D_feedthrough_vector", shape=(self.model_dimension,), 
            initializer="zeros", trainable=True, dtype=variable_dtype_for_weights
        )

        # Ensure sub-layers are built (can also be done in first call)
        if not self.B_project_to_rank_dense.built: self.B_project_to_rank_dense.build(tf.TensorShape((None, None, self.model_dimension)))
        if not self.B_expand_to_state_dense.built: self.B_expand_to_state_dense.build(tf.TensorShape((None, None, self.s4_rank)))
        if self.C_project_to_rank_dense and not self.C_project_to_rank_dense.built: self.C_project_to_rank_dense.build(tf.TensorShape((None, None, self.internal_state_dimension_N)))
        if not self.C_expand_to_model_dense.built: self.C_expand_to_model_dense.build(tf.TensorShape((None, None, self.s4_rank)))
        
        super().build(input_tensor_shape_ignored)

    def _apply_butterfly_structured_matrix_A(self, state_tensor_S_input):
        if self.num_butterfly_levels == 0 or self.internal_state_dimension_N == 0: return state_tensor_S_input
        
        current_tensor_shape = tf.shape(state_tensor_S_input)
        batch_size_val, sequence_length_val = current_tensor_shape[0], current_tensor_shape[1]
        
        reshaped_tensor_for_butterfly = tf.reshape(state_tensor_S_input, tf.stack([batch_size_val, sequence_length_val] + [2] * self.num_butterfly_levels))
        rank_of_reshaped_tensor = 2 + self.num_butterfly_levels # Batch, Seq, plus N levels of 2s

        current_state_being_transformed = reshaped_tensor_for_butterfly
        for level_index_val in range(self.num_butterfly_levels):
            level_A_factor_matrix = tf.cast(self.A_butterfly_factors_list[level_index_val], current_state_being_transformed.dtype)
            axis_for_current_level_processing = 2 + level_index_val # 0-indexed from batch dim
            
            # Permute to bring the current processing axis to the end for matmul
            permutation_order_list = [0, 1] + \
                                     [axis_idx for axis_idx in range(2, rank_of_reshaped_tensor) if axis_idx != axis_for_current_level_processing] + \
                                     [axis_for_current_level_processing]
            inverse_permutation_order_list = tf.math.invert_permutation(tf.constant(permutation_order_list, tf.int32))
            
            permuted_tensor_for_matmul = tf.transpose(current_state_being_transformed, perm=permutation_order_list)
            shape_of_permuted_tensor = tf.shape(permuted_tensor_for_matmul)
            
            # Flatten all dimensions except the last one (which is size 2)
            flattened_dimension_size_for_matmul = tf.reduce_prod(shape_of_permuted_tensor[:-1])
            flattened_tensor_input_to_matmul = tf.reshape(permuted_tensor_for_matmul, [flattened_dimension_size_for_matmul, 2])
            
            multiplied_state_tensor = tf.matmul(flattened_tensor_input_to_matmul, level_A_factor_matrix)
            
            reshaped_state_after_multiplication = tf.reshape(multiplied_state_tensor, shape_of_permuted_tensor)
            current_state_being_transformed = tf.transpose(reshaped_state_after_multiplication, perm=inverse_permutation_order_list)
            
        return tf.reshape(current_state_being_transformed, tf.stack([batch_size_val, sequence_length_val, self.internal_state_dimension_N]))

    def call(self, input_sequence_tensor_u):
        projected_B_to_rank_tensor = self.B_project_to_rank_dense(input_sequence_tensor_u)
        expanded_B_to_state_tensor = self.B_expand_to_state_dense(projected_B_to_rank_tensor)
        
        state_tensor_after_A_transform = self._apply_butterfly_structured_matrix_A(expanded_B_to_state_tensor)
        
        if self.internal_state_dimension_N > 0 and self.C_project_to_rank_dense is not None:
            projected_C_to_rank_tensor = self.C_project_to_rank_dense(state_tensor_after_A_transform)
        else:
            shape_of_input_u = tf.shape(input_sequence_tensor_u)
            projected_C_to_rank_tensor = tf.zeros((shape_of_input_u[0], shape_of_input_u[1], self.s4_rank), dtype=input_sequence_tensor_u.dtype)
            
        output_C_to_model_dim_tensor = self.C_expand_to_model_dense(projected_C_to_rank_tensor)
        output_D_feedthrough_path = tf.cast(self.D_feedthrough_vector_weight, input_sequence_tensor_u.dtype) * input_sequence_tensor_u
        
        return output_C_to_model_dim_tensor + output_D_feedthrough_path

    def get_config(self):
        config = super().get_config()
        config.update({
            "model_dimension": self.model_dimension, 
            "state_dimension_config": self.configured_state_dimension, # Original value
            "ssm_rank_config": self.s4_rank # This now correctly refers to the actual rank value
        })
        return config

class OutputHead(keras.Model): 
    def __init__(self, shared_embedding_layer_ref, vocabulary_size, name="OutputHead_default", **kwargs):
        super().__init__(name=name, **kwargs) 
        self.layer_norm = LayerNormalization(epsilon=1e-6, name=f"{self.name}_layer_norm")
        self.shared_embedding_layer_ref = shared_embedding_layer_ref # Store reference
        self.lambda_cast_to_float32 = Lambda(lambda x_tensor: tf.cast(x_tensor, tf.float32), name=f"{self.name}_cast_to_float32")
        self.softmax_activation = Activation('softmax', name=f"{self.name}_final_softmax")

    def call(self, input_hidden_states_tensor):
        normalized_hidden_states = self.layer_norm(input_hidden_states_tensor)
        output_logits_tensor = self.shared_embedding_layer_ref.compute_logits(normalized_hidden_states)
        logits_as_float32_tensor = self.lambda_cast_to_float32(output_logits_tensor)
        probabilities_output = self.softmax_activation(logits_as_float32_tensor)
        return probabilities_output

    def get_config(self):
        config = super().get_config()
        config.update({"vocabulary_size": self.shared_embedding_layer_ref.vocab_size})
        return config

class Conv2DBlock(keras.Model):
    def __init__(self, output_embedding_dimension, name="Conv2DBlock_default", **kwargs): # Name provided by caller
        super().__init__(name=name, **kwargs)
        self.output_embedding_dimension = output_embedding_dimension
        
        self.expand_dims_norm_input = ExpandDimsLayer(axis=-1, name=f"{self.name}_expand_norm_input")
        self.expand_dims_fourier_input = ExpandDimsLayer(axis=-1, name=f"{self.name}_expand_fourier_input")
        self.concatenate_inputs_layer = Concatenate(axis=-1, name=f"{self.name}_concatenate_inputs")
        
        self.convolution_path_layers_list = []
        # Configuration for the Conv2D layers in the block
        filters_config_list = [64, 128, 1] 
        kernels_config_list = [(7,7), (5,5), (3,3)] 
        activations_config_list = [tf.nn.swish, tf.nn.swish, None]

        for i, (num_filters_val, kernel_dims_val, activation_fn_val) in enumerate(zip(filters_config_list, kernels_config_list, activations_config_list)):
            self.convolution_path_layers_list.append(
                CausalPadding2D(kernel_dims_val, name=f"{self.name}_causal_padding_2d_{i}")
            )
            self.convolution_path_layers_list.append(
                Conv2D(
                    filters=num_filters_val, kernel_size=kernel_dims_val, padding='valid',
                    activation=activation_fn_val, name=f"{self.name}_conv2d_layer_{i}"
                )
            )
            
        self.squeeze_channel_dim_layer = SqueezeLayer(axis=-1, name=f"{self.name}_squeeze_channel_dim")
        self.projection_to_output_dim_dense_layer = Dense(units=output_embedding_dimension, name=f"{self.name}_projection_to_output_dim_dense")
        self.time_distributed_final_projection_layer = TimeDistributed(self.projection_to_output_dim_dense_layer, name=f"{self.name}_time_distributed_projection")

    def call(self, inputs_list_args): 
        normalized_sequence_input, fourier_features_input = inputs_list_args
        
        expanded_normalized_seq = self.expand_dims_norm_input(normalized_sequence_input)
        expanded_fourier_features = self.expand_dims_fourier_input(fourier_features_input)
        
        concatenated_conv_input = self.concatenate_inputs_layer([expanded_normalized_seq, expanded_fourier_features])
        
        current_tensor_in_convolution_path = concatenated_conv_input
        for layer_in_path in self.convolution_path_layers_list:
            current_tensor_in_convolution_path = layer_in_path(current_tensor_in_convolution_path)
        
        squeezed_output_from_conv_path = self.squeeze_channel_dim_layer(current_tensor_in_convolution_path)
        final_projected_output_sequence = self.time_distributed_final_projection_layer(squeezed_output_from_conv_path)
        return final_projected_output_sequence

    def get_config(self):
        config = super().get_config()
        config.update({"output_embedding_dimension": self.output_embedding_dimension})
        return config

# =========================================================================================
# =          G3Block and MTPG3Block (Main Architectural Components)                     =
# =========================================================================================
def G3Block(
    embedding_dimension_val, shared_memory_module_inst, shared_fourier_layer_inst, 
    ssm_rank_val, ssm_state_dimension_multiplier_val, layer_idx, block_prefix_str="G3Block_"
): 
    current_block_name = f"{block_prefix_str}{layer_idx}"
    
    input_sequence_tensor = Input(shape=(None, embedding_dimension_val), name=f"{current_block_name}_input_sequence_tensor")
    input_memory_tensor = Input(shape=(NUM_REGISTERS, MEMORY_DIM), name=f"{current_block_name}_input_memory_tensor", dtype=shared_memory_module_inst.compute_dtype)
    
    residual_connection_tensor = input_sequence_tensor
    normalized_sequence_tensor = LayerNormalization(epsilon=1e-6, name=f"{current_block_name}_input_layer_norm")(input_sequence_tensor)
    
    depthwise_conv_kernel_size = 4
    padded_sequence_for_depthwise = CausalPadding1D(depthwise_conv_kernel_size, name=f"{current_block_name}_depthwise_causal_padding")(normalized_sequence_tensor)
    depthwise_conv_output_tensor = DepthwiseConv1D(
        kernel_size=depthwise_conv_kernel_size, padding='valid', activation=tf.nn.swish, 
        depth_multiplier=1, name=f"{current_block_name}_depthwise_conv_operation"
    )(padded_sequence_for_depthwise)
    sequence_after_depthwise_conv = Add()([normalized_sequence_tensor, depthwise_conv_output_tensor])
    
    features_for_main_branches = LayerNormalization(epsilon=1e-6, name=f"{current_block_name}_features_layer_norm")(sequence_after_depthwise_conv)
    fourier_features_for_conv2d = shared_fourier_layer_inst(normalized_sequence_tensor) # Note: Using original normalized_sequence_tensor
    
    ssm_core_output_tensor = SSMCore(
        model_dimension=embedding_dimension_val, 
        state_dimension_config=embedding_dimension_val * ssm_state_dimension_multiplier_val, # Corrected state_dimension to state_dimension_config
        ssm_rank_config=ssm_rank_val, 
        name=f"{current_block_name}_ssm"
    )(features_for_main_branches)
    
    conv2d_block_output_tensor = Conv2DBlock(
        output_embedding_dimension=embedding_dimension_val, 
        name=f"{current_block_name}_conv2d_block" 
    )([features_for_main_branches, fourier_features_for_conv2d])
    
    combined_ssm_and_conv2d_tensor = Add()([ssm_core_output_tensor, conv2d_block_output_tensor])
    
    lambda_for_slicing_last_token = Lambda(lambda x_input_tensor: x_input_tensor[:, -1:, :], name=f"{current_block_name}_slice")
    query_tensor_for_memory_read = lambda_for_slicing_last_token(combined_ssm_and_conv2d_tensor)
    
    read_signal_vector_from_memory, _ = shared_memory_module_inst([query_tensor_for_memory_read, input_memory_tensor], mode='read')
    
    lambda_for_tiling_read_signal = Lambda(
        lambda list_of_inputs: tf.tile(tf.cast(list_of_inputs[0], list_of_inputs[1].dtype), [1, tf.shape(list_of_inputs[1])[1], 1]), 
        name=f"{current_block_name}_lambda_tile_read_signal"
    )
    tiled_read_signal_sequence_tensor = lambda_for_tiling_read_signal([read_signal_vector_from_memory, combined_ssm_and_conv2d_tensor])
    
    sequence_tensor_after_memory_read = Add()([combined_ssm_and_conv2d_tensor, tiled_read_signal_sequence_tensor])
    
    query_tensor_for_memory_write = lambda_for_slicing_last_token(sequence_tensor_after_memory_read)
    _, updated_memory_state_tensor = shared_memory_module_inst([query_tensor_for_memory_write, input_memory_tensor], mode='write')
    
    output_sequence_before_pooling_tensor = Add(name=f"{current_block_name}_add_final_residual_connection")([residual_connection_tensor, sequence_tensor_after_memory_read])
    
    pooled_output_sequence_tensor = ConditionalAveragePooling1D(
        pool_size=2, strides=2, padding_if_pool='SAME', name=f"{current_block_name}_final_conditional_avg_pooling"
    )(output_sequence_before_pooling_tensor)
    
    return keras.Model(
        inputs=[input_sequence_tensor, input_memory_tensor], 
        outputs=[pooled_output_sequence_tensor, updated_memory_state_tensor], 
        name=current_block_name
    )

def MTPG3Block(
    embedding_dimension_val, shared_memory_module_inst, shared_fourier_layer_inst, 
    ssm_rank_val, ssm_state_dimension_multiplier_val, layer_idx, block_prefix_str="MTPG3Block_" ):
    current_block_name = f"{block_prefix_str}{layer_idx}"
    
    input_sequence_tensor = Input(shape=(None, embedding_dimension_val), name=f"{current_block_name}_input_sequence_tensor")
    input_memory_tensor = Input(shape=(NUM_REGISTERS, MEMORY_DIM), name=f"{current_block_name}_input_memory_tensor", dtype=shared_memory_module_inst.compute_dtype)
    
    residual_connection_tensor = input_sequence_tensor
    normalized_sequence_tensor = LayerNormalization(epsilon=1e-6, name=f"{current_block_name}_input_layer_norm")(input_sequence_tensor)
    
    depthwise_conv_kernel_size = 4
    padded_sequence_for_depthwise = CausalPadding1D(depthwise_conv_kernel_size, name=f"{current_block_name}_depthwise_causal_padding")(normalized_sequence_tensor)
    depthwise_conv_output_tensor = DepthwiseConv1D(
        kernel_size=depthwise_conv_kernel_size, padding='valid', activation=tf.nn.swish, 
        depth_multiplier=1, name=f"{current_block_name}_depthwise_conv_op"
    )(padded_sequence_for_depthwise)
    sequence_after_dw_residual = Add(name=f"{current_block_name}_add_residual_after_dw")([normalized_sequence_tensor, depthwise_conv_output_tensor])
    
    features_for_main_branches = LayerNormalization(epsilon=1e-6, name=f"{current_block_name}_features_layer_norm")(sequence_after_dw_residual)
    fourier_features_for_conv2d = shared_fourier_layer_inst(normalized_sequence_tensor) # Note: Using original normalized_sequence_tensor

    ssm_core_output_tensor = SSMCore(
        model_dimension=embedding_dimension_val, 
        state_dimension_config=embedding_dimension_val * ssm_state_dimension_multiplier_val, # Corrected state_dimension to state_dimension_config
        ssm_rank_config=ssm_rank_val, 
        name=f"{current_block_name}_ssm"
    )(features_for_main_branches)

    conv2d_block_output_tensor = Conv2DBlock(
        output_embedding_dimension=embedding_dimension_val,
        name=f"{current_block_name}_conv2d_block" 
    )([features_for_main_branches, fourier_features_for_conv2d])
    
    combined_ssm_and_conv2d_tensor = Add()([ssm_core_output_tensor, conv2d_block_output_tensor])
    
    lambda_slice_last_token = Lambda(lambda x_tensor: x_tensor[:, -1:, :], name=f"{current_block_name}_slice")
    memory_read_query_tensor = lambda_slice_last_token(combined_ssm_and_conv2d_tensor)
    read_signal_vector_from_memory, _ = shared_memory_module_inst([memory_read_query_tensor, input_memory_tensor], mode='read')
    
    lambda_for_tiling_read_signal = Lambda(
        lambda input_list_tile: tf.tile(tf.cast(input_list_tile[0], input_list_tile[1].dtype), [1, tf.shape(input_list_tile[1])[1], 1]), 
        name=f"{current_block_name}_lambda_tile_read_signal"
    )
    tiled_read_signal_sequence_tensor = lambda_for_tiling_read_signal([read_signal_vector_from_memory, combined_ssm_and_conv2d_tensor])
    
    sequence_tensor_after_memory_read = Add()([combined_ssm_and_conv2d_tensor, tiled_read_signal_sequence_tensor])
    
    memory_write_query_tensor = lambda_slice_last_token(sequence_tensor_after_memory_read)
    _, updated_memory_output_state = shared_memory_module_inst([memory_write_query_tensor, input_memory_tensor], mode='write')
    
    # Final Residual Connection - MTPG3Block does NOT pool its output sequence
    output_sequence_tensor = Add(name=f"{current_block_name}_res")([residual_connection_tensor, sequence_tensor_after_memory_read])
    
    return keras.Model(
        inputs=[input_sequence_tensor, input_memory_tensor], 
        outputs=[output_sequence_tensor, updated_memory_output_state], 
        name=current_block_name
    )

# =========================================================================================
# =                                 Main Model Definition
# =========================================================================================
def ExpG3_build_model(eject_mtp_path_flag=False): 
    print(f"\n--- Building ExpG3 Model (MTP Path Active: {not eject_mtp_path_flag and NUM_MTP_HEADS > 0}) ---")
    print(f"Configuration: Vocab Size={ACTUAL_VOCAB_SIZE}, Embedding Dim={EMBEDDING_DIM}, Max Sequence Length={MAX_SEQ_LEN}, "
          f"Number of G3 Layers={NUM_MODEL_LAYERS}, Number of MTP Heads={NUM_MTP_HEADS}, SSM State Multiplier={SSM_STATE_DIM_MULTIPLIER}")

    token_ids_input_layer = Input(shape=(MAX_SEQ_LEN,), dtype="int32", name="token_ids_input_layer")
    initial_memory_input_layer = Input(shape=(NUM_REGISTERS, MEMORY_DIM), dtype="float32", name="initial_memory_state_input_layer")

    # Shared Layers Instantiation
    shared_autonway_embedding_layer = Embedding(ACTUAL_VOCAB_SIZE,EMBEDDING_DIM)
    shared_fused_positional_encoding_layer = FusedPositionalEncoding(EMBEDDING_DIM,MAX_SEQ_LEN,name="shared_fused_positional_encoding_layer")
    shared_fourier_features_gen_layer = FourierFeatures(EMBEDDING_DIM,name="shared_fourier_features_generator_layer")
    shared_stateful_memory_module_inst = StatefulMemoryModule(NUM_REGISTERS,MEMORY_DIM,name="shared_stateful_memory_module_instance")
    
    # Initial Embedding and Positional Encoding
    sequence_token_embeddings = shared_autonway_embedding_layer(token_ids_input_layer)
    sequence_embeddings_with_position = shared_fused_positional_encoding_layer(sequence_token_embeddings)

    # Cast initial memory to the compute dtype expected by the memory module
    casted_initial_memory_for_module = Lambda(
        lambda memory_tensor: tf.cast(memory_tensor, shared_stateful_memory_module_inst.compute_dtype),
        name="lambda_cast_initial_memory_for_module"
    )(initial_memory_input_layer)

    # Main Processing Path (Stacked G3Blocks)
    current_sequence_output = sequence_embeddings_with_position
    current_memory_state_output = casted_initial_memory_for_module
    calculated_ssm_rank = EMBEDDING_DIM // SSM_RANK_DIVISOR
    """
    for layer_idx_val in range(NUM_MODEL_LAYERS): # Changed loop variable name for clarity
        g3_block_module = G3Block(
            embedding_dimension_val=EMBEDDING_DIM,
            shared_memory_module_inst=shared_stateful_memory_module_inst,
            shared_fourier_layer_inst=shared_fourier_features_gen_layer,
            ssm_rank_val=calculated_ssm_rank,
            ssm_state_dimension_multiplier_val=SSM_STATE_DIM_MULTIPLIER,
            layer_idx=str(layer_idx_val) # Pass layer_idx_val as string for block name
        )
        current_sequence_output, current_memory_state_output = g3_block_module([current_sequence_output, current_memory_state_output])
    """
    main_path_final_sequence_output = current_sequence_output
    main_path_final_memory_output = current_memory_state_output
    
    # List to store outputs from all prediction heads
    list_of_all_prediction_outputs = [
        OutputHead(
            shared_embedding_layer_ref=shared_autonway_embedding_layer, 
            vocabulary_size=ACTUAL_VOCAB_SIZE, 
            name="OutputHead_MainPath" # Provide full unique name
        )(main_path_final_sequence_output)
    ]
    
    # --- Multi-Task Prediction (MTP) Path ---
    # This path starts from the output of the main G3 processing stack
    last_processed_sequence_for_mtp = main_path_final_sequence_output
    last_processed_memory_for_mtp = main_path_final_memory_output

    # Conditionally build MTP path
    should_build_mtp_path = not eject_mtp_path_flag and NUM_MTP_HEADS > 0
    if should_build_mtp_path:
        for mtp_head_idx in range(NUM_MTP_HEADS):
            mtp_g3_block_module = MTPG3Block( # MTPG3Block does not have internal pooling
                embedding_dimension_val=EMBEDDING_DIM,
                shared_memory_module_inst=shared_stateful_memory_module_inst,
                shared_fourier_layer_inst=shared_fourier_features_gen_layer,
                ssm_rank_val=calculated_ssm_rank,
                ssm_state_dimension_multiplier_val=SSM_STATE_DIM_MULTIPLIER,
                layer_idx=str(mtp_head_idx) # CORRECTED: Use 'layer_idx' to match MTPG3Block definition
            )
            # Output of previous MTP block (or main path for first MTP) becomes input to current
            last_processed_sequence_for_mtp, last_processed_memory_for_mtp = mtp_g3_block_module([last_processed_sequence_for_mtp, last_processed_memory_for_mtp])
            
            list_of_all_prediction_outputs.append(
                OutputHead(
                    shared_embedding_layer_ref=shared_autonway_embedding_layer, 
                    vocabulary_size=ACTUAL_VOCAB_SIZE, 
                    name=f"OutputHead_MTP_{mtp_head_idx}" # Explicit full name
                )(last_processed_sequence_for_mtp)
            )
            
    # The final memory state to be output by the model is the one from the last processed block
    final_memory_state_from_last_block = last_processed_memory_for_mtp # This will be main_path_mem_output if MTP is skipped
    
    final_memory_output_casted_to_float32 = Lambda(
        lambda memory_tensor: tf.cast(memory_tensor, tf.float32),
        name="cast_f32"
    )(final_memory_state_from_last_block)
    
    return keras.Model(
        inputs=[token_ids_input_layer, initial_memory_input_layer],
        outputs=[list_of_all_prediction_outputs, final_memory_output_casted_to_float32],
        name="ExpG3_Model"
    )

# =========================================================================================
# =                             TPU Strategy and Utilities
# =========================================================================================
def get_tpu_strategy():
    global _TPU_INITIALIZED_IN_SESSION, _LAST_TPU_RESOLVER 
    current_strategy_object = None
    tpu_system_is_ready = False
    print("Attempting to initialize TPU strategy...")

    if _TPU_INITIALIZED_IN_SESSION and _LAST_TPU_RESOLVER:
        print(f"TPU system was previously initialized with resolver: {_LAST_TPU_RESOLVER.master()}. Attempting to re-create TPUStrategy object.")
        try:
            current_strategy_object = tf.distribute.TPUStrategy(_LAST_TPU_RESOLVER)
            print(f"TPUStrategy re-created successfully. Num replicas: {current_strategy_object.num_replicas_in_sync}")
            tpu_system_is_ready = True 
        except Exception as error_recreating_strategy_object:
            print(f"Failed to re-create TPUStrategy from previous resolver: {error_recreating_strategy_object}\n{traceback.format_exc()}")
            _TPU_INITIALIZED_IN_SESSION = False 
            _LAST_TPU_RESOLVER = None
            tpu_system_is_ready = False 
    
    if not tpu_system_is_ready: 
        _TPU_INITIALIZED_IN_SESSION = False 
        _LAST_TPU_RESOLVER = None
        try:
            physical_devices = tf.config.list_physical_devices()
            print(f"Available physical devices: {physical_devices}")
            if not any(device.device_type == 'TPU' for device in physical_devices):
                print("No TPU physical devices detected by TensorFlow.")
                # Do not raise RuntimeError here, let it fallback.
                # raise RuntimeError("No TPU hardware found at the physical device level.")
        except Exception as error_listing_devices:
            print(f"Could not list physical devices or no TPUs found: {error_listing_devices}")
            # Proceed to fallback without explicit return here, let the logic flow to fallback
        
        # Try TPU initialization only if TPU physical devices were potentially found or listing failed non-critically
        if any(device.device_type == 'TPU' for device in tf.config.list_physical_devices()):
            try:
                print("\nAttempt 1 (Fresh Initialization): Using TPUClusterResolver.connect()")
                resolver_from_connect = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
                print(f"TPUClusterResolver.connect() successful. Resolver: {resolver_from_connect}, Master: {resolver_from_connect.master()}")
                print("Initializing TPU system (Attempt 1 via connect)...")
                tf.tpu.experimental.initialize_tpu_system(resolver_from_connect)
                print("TPU system initialized successfully via connect().")
                _TPU_INITIALIZED_IN_SESSION = True
                _LAST_TPU_RESOLVER = resolver_from_connect
                tpu_system_is_ready = True
            except Exception as error_connect_attempt:
                print(f"TPUClusterResolver.connect() attempt failed: {error_connect_attempt}\n{traceback.format_exc()}")
            
            if not tpu_system_is_ready:
                print("\nAttempt 2 (Fresh Initialization): Manual TPUClusterResolver with 'local' or ENV VARS.")
                resolver_manual_attempt = None
                tpu_identifier_for_manual_attempt = None
                try:
                    tpu_identifier_for_manual_attempt = os.environ.get('TPU_NAME') or os.environ.get('KAGGLE_TPU_NAME') or 'local'
                    print(f"Using TPU identifier for manual attempt: '{tpu_identifier_for_manual_attempt}'")
                    resolver_manual_attempt = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_identifier_for_manual_attempt)
                    print(f"Instantiated TPUClusterResolver. Master before connect_to_cluster: {resolver_manual_attempt.master()}")
                    print(f"Attempting tf.config.experimental_connect_to_cluster for tpu='{tpu_identifier_for_manual_attempt}'...")
                    tf.config.experimental_connect_to_cluster(resolver_manual_attempt)
                    print(f"tf.config.experimental_connect_to_cluster successful. Master after connect: {resolver_manual_attempt.master()}")
                    print("Initializing TPU system (Attempt 2 via manual resolver)...")
                    tf.tpu.experimental.initialize_tpu_system(resolver_manual_attempt)
                    print("TPU system initialized successfully via manual resolver.")
                    _TPU_INITIALIZED_IN_SESSION = True
                    _LAST_TPU_RESOLVER = resolver_manual_attempt
                    tpu_system_is_ready = True
                except Exception as error_manual_init_attempt:
                    print(f"Attempt 2 with tpu='{tpu_identifier_for_manual_attempt}' failed: {error_manual_init_attempt}\n{traceback.format_exc()}")
        
        if tpu_system_is_ready: # If system was initialized successfully by either fresh attempt
            try:
                current_strategy_object = tf.distribute.TPUStrategy(_LAST_TPU_RESOLVER) # Use the resolver that worked
                print(f"TPUStrategy object created successfully. Number of replicas: {current_strategy_object.num_replicas_in_sync}")
            except Exception as error_creating_strategy_object_after_init:
                print(f"Failed to create TPUStrategy object even after successful system initialization: {error_creating_strategy_object_after_init}\n{traceback.format_exc()}")
                current_strategy_object = get_cpu_gpu_fallback_strategy() 
        else: # All fresh TPU initialization attempts failed or no TPUs detected
            current_strategy_object = get_cpu_gpu_fallback_strategy()
            
    if current_strategy_object is None: 
        print("CRITICAL: Strategy object is still None after all attempts and fallbacks. Defaulting to OneDeviceStrategy on CPU.")
        current_strategy_object = tf.distribute.OneDeviceStrategy(device="/cpu:0")

    print(f"\nSelected strategy for training: {type(current_strategy_object).__name__}")
    num_replicas_final = getattr(current_strategy_object, 'num_replicas_in_sync', 1) 
    print(f"Number of replicas in sync for the selected strategy: {num_replicas_final}")
    return current_strategy_object

def get_cpu_gpu_fallback_strategy():
    print("\nFalling back to CPU/GPU strategy as TPU initialization failed or no TPUs available.")
    strategy_object_fallback = None
    gpu_physical_devices = tf.config.list_physical_devices('GPU')
    if gpu_physical_devices:
        print(f"Found {len(gpu_physical_devices)} physical GPU(s): {gpu_physical_devices}")
        try:
            for gpu_device_instance in gpu_physical_devices:
                tf.config.experimental.set_memory_growth(gpu_device_instance, True)
            strategy_object_fallback = tf.distribute.MirroredStrategy()
            print(f"Using MirroredStrategy with {strategy_object_fallback.num_replicas_in_sync} GPU(s).")
        except RuntimeError as error_gpu_initialization:
            print(f"Error initializing GPUs for MirroredStrategy: {error_gpu_initialization}. Defaulting to CPU strategy.")
            strategy_object_fallback = tf.distribute.OneDeviceStrategy(device="/cpu:0")
    else:
        print("No GPUs found. Using default OneDeviceStrategy on CPU.")
        strategy_object_fallback = tf.distribute.OneDeviceStrategy(device="/cpu:0")
    return strategy_object_fallback

# =========================================================================================
# =                        Data Pipeline (FineWeb-Edu from HF)
# =========================================================================================
def create_dataset_from_hf_stream(
    hf_dataset_name_str, hf_split_name_str, tokenizer_instance_obj, global_tf_data_batch_size_val,
    max_sequence_length_val, num_mtp_heads_val, eos_token_id_val
):
    print(f"Creating tf.data.Dataset from Hugging Face stream: {hf_dataset_name_str}/{hf_split_name_str}")
    hugging_face_iterable_dataset = load_dataset(hf_dataset_name_str,streaming=True,split=hf_split_name_str,trust_remote_code=True)
    
    num_total_target_tokens_val = 1 + num_mtp_heads_val 
    chunk_length_for_generator = max_sequence_length_val + num_total_target_tokens_val
    print(f"Using chunk length for dataset generator: {chunk_length_for_generator} "
          f"(max_seq_len={max_sequence_length_val} + {num_total_target_tokens_val} target tokens)")

    def hugging_face_token_generator_function(): 
        token_buffer_list = []
        for example_dictionary in hugging_face_iterable_dataset:
            text_from_example = example_dictionary.get("text", "")
            if text_from_example: 
                token_ids_list = tokenizer_instance_obj.encode(text_from_example, add_special_tokens=False) + [eos_token_id_val]
                token_buffer_list.extend(token_ids_list)
                while len(token_buffer_list) >= chunk_length_for_generator:
                    yield token_buffer_list[:chunk_length_for_generator]
                    token_buffer_list = token_buffer_list[chunk_length_for_generator:]

    tensorflow_dataset_object = tf.data.Dataset.from_generator(
        hugging_face_token_generator_function,
        output_signature=tf.TensorSpec(shape=(chunk_length_for_generator,), dtype=tf.int32) 
    )

    def preprocess_tensorflow_dataset_chunk(chunk_tensor_input_slice):
        input_ids_for_model_slice = chunk_tensor_input_slice[0:max_sequence_length_val]
        list_of_target_tokens_for_heads = []
        for i in range(num_total_target_tokens_val):
            target_token_value = chunk_tensor_input_slice[max_sequence_length_val + i]
            list_of_target_tokens_for_heads.append(tf.reshape(target_token_value, (1,)))
        
        initial_memory_state_tensor = tf.zeros((NUM_REGISTERS, MEMORY_DIM), dtype=tf.float32)
        return (input_ids_for_model_slice, initial_memory_state_tensor), tuple(list_of_target_tokens_for_heads)

    tensorflow_dataset_object = tensorflow_dataset_object.map(preprocess_tensorflow_dataset_chunk, num_parallel_calls=tf.data.AUTOTUNE)
    tensorflow_dataset_object = tensorflow_dataset_object.shuffle(buffer_size=1024, reshuffle_each_iteration=True) 
    tensorflow_dataset_object = tensorflow_dataset_object.batch(global_tf_data_batch_size_val, drop_remainder=True) 
    tensorflow_dataset_object = tensorflow_dataset_object.prefetch(tf.data.AUTOTUNE)
    print("Hugging Face streaming dataset pipeline created and configured successfully.")
    return tensorflow_dataset_object

# =========================================================================================
# =                             Training Loop & LR Schedule
# =========================================================================================
class CustomLearningRateSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak_learning_rate_val, warmup_steps_val, total_decay_steps_val, end_learning_factor_val=0.1, name=None):
        super().__init__()
        self.peak_lr = tf.cast(peak_learning_rate_val, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps_val, tf.float32)
        self.total_decay_steps = tf.maximum(tf.cast(total_decay_steps_val, tf.float32), 1.0) # Ensure at least 1
        self.end_lr_factor = tf.cast(end_learning_factor_val, tf.float32)
        self.name = name

    def __call__(self, step_count_tensor):
        step_float_tensor = tf.cast(step_count_tensor, tf.float32)
        
        learning_rate_in_warmup = (step_float_tensor / self.warmup_steps) * self.peak_lr
        
        decay_phase_progress_ratio = tf.clip_by_value((step_float_tensor - self.warmup_steps) / self.total_decay_steps, 0.0, 1.0)
        cosine_decay_value = 0.5 * (1.0 + tf.cos(tf.constant(np.pi, dtype=tf.float32) * decay_phase_progress_ratio))
        
        target_final_learning_rate = self.peak_lr * self.end_lr_factor
        learning_rate_in_decay = (self.peak_lr - target_final_learning_rate) * cosine_decay_value + target_final_learning_rate
        
        is_currently_in_warmup = step_float_tensor < self.warmup_steps
        return tf.cond(is_currently_in_warmup, lambda: learning_rate_in_warmup, lambda: learning_rate_in_decay)

    def get_config(self):
        return {
            "peak_learning_rate": float(self.peak_lr.numpy()),
            "warmup_steps": float(self.warmup_steps.numpy()),
            "total_decay_steps": float(self.total_decay_steps.numpy()),
            "end_learning_factor": float(self.end_lr_factor.numpy()),
            "name": self.name
        }

# =========================================================================================
# =                             Text Generation Utilities
# =========================================================================================
def generate_text_from_model(
    # tpu_strategy_arg unused if model called directly
    tpu_strategy_arg, # This argument is passed but not directly used in the function body for model inference
    keras_model_to_generate_from, 
    tokenizer_to_use, 
    input_prompt_string,
    initial_memory_state_numpy, 
    max_tokens_to_generate=60, 
    text_sampling_temperature=0.8 
):
    print(f"\n--- Generating text for prompt: '{input_prompt_string}' ---")
    
    current_token_ids_list = tokenizer_to_use.encode(input_prompt_string, add_special_tokens=False)
    all_generated_token_ids = list(current_token_ids_list)

    if len(initial_memory_state_numpy.shape) == 2: 
        current_batched_memory_np = np.expand_dims(initial_memory_state_numpy, axis=0)
    elif len(initial_memory_state_numpy.shape) == 3: 
        current_batched_memory_np = initial_memory_state_numpy
    else:
        raise ValueError(f"Initial memory state numpy array has an unexpected shape: {initial_memory_state_numpy.shape}")
    
    current_memory_state_tf_tensor = tf.convert_to_tensor(current_batched_memory_np, dtype=tf.float32)

    for _ in range(max_tokens_to_generate):
        padded_input_ids_np_array = np.full((1, MAX_SEQ_LEN), PAD_TOKEN_ID, dtype=np.int32)
        
        tokens_in_current_window = all_generated_token_ids[-MAX_SEQ_LEN:]
        num_tokens_in_window = len(tokens_in_current_window)
        padded_input_ids_np_array[0, :num_tokens_in_window] = tokens_in_current_window
        
        input_ids_tf_tensor_for_model = tf.convert_to_tensor(padded_input_ids_np_array, dtype=tf.int32)
        
        # Model call is direct, not via strategy.run() for generation.
        list_of_prediction_head_outputs, next_memory_state_tf_tensor = keras_model_to_generate_from(
            [input_ids_tf_tensor_for_model, current_memory_state_tf_tensor], training=False
        )
        current_memory_state_tf_tensor = next_memory_state_tf_tensor 

        # Assume the last prediction head's output is the primary one for generation
        logits_tensor_from_final_pred_head = list_of_prediction_head_outputs[-1] 
        
        # The logits for the *next* token are at the position corresponding to the *last input* token
        index_for_next_token_logit_slice = min(num_tokens_in_window - 1, tf.shape(logits_tensor_from_final_pred_head)[1] - 1)
        if index_for_next_token_logit_slice < 0: # Handle empty prompt case if that was allowed (it's not here)
            index_for_next_token_logit_slice = 0 
        
        next_token_logits_tf_slice = logits_tensor_from_final_pred_head[0, index_for_next_token_logit_slice, :]
        
        if text_sampling_temperature == 0: # Argmax sampling
            generated_id_for_next_token = tf.argmax(next_token_logits_tf_slice, axis=-1).numpy()
        else: # Temperature sampling
            scaled_logits_for_sampling_tensor = next_token_logits_tf_slice / text_sampling_temperature
            generated_id_for_next_token = tf.random.categorical(tf.expand_dims(scaled_logits_for_sampling_tensor, axis=0), num_samples=1)[0, 0].numpy()
        
        if generated_id_for_next_token == EOS_TOKEN_ID:
            print(f"<EOS token generated. Halting generation.>")
            break
        
        all_generated_token_ids.append(int(generated_id_for_next_token))
        
    full_generated_text_string_output = tokenizer_to_use.decode(all_generated_token_ids, skip_special_tokens=True)
    print(f"Generated text (total {len(all_generated_token_ids)} tokens): '{full_generated_text_string_output}'")
    return full_generated_text_string_output

# =========================================================================================
# =                                     Main Function
# =========================================================================================
def main(total_train_steps_override=None):
    # cleanup_tensorflow_runtimes() # Call if re-running main() in the same interactive session

    global TOTAL_TRAIN_STEPS_CONFIG, tokenizer 
    if total_train_steps_override is not None:
        current_total_train_steps = total_train_steps_override
        print(f"Overriding TOTAL_TRAIN_STEPS for this run to: {current_total_train_steps}")
    else:
        current_total_train_steps = TOTAL_TRAIN_STEPS_CONFIG
    
    if tokenizer is None: 
        initialize_tokenizer()
        if tokenizer is None: 
            print("CRITICAL: Tokenizer initialization failed in main. Exiting.")
            return 
    
    print(f"TensorFlow Version: {tf.__version__}, Keras Version: {keras.__version__}")
    
    distributed_strategy_instance = get_tpu_strategy()
    num_replicas_in_sync_val = getattr(distributed_strategy_instance, 'num_replicas_in_sync', 1)
    global_batch_size_val = BATCH_SIZE_PER_REPLICA * num_replicas_in_sync_val
    
    print(f"Global Batch Size: {global_batch_size_val} = {BATCH_SIZE_PER_REPLICA} (per replica) * {num_replicas_in_sync_val} (replicas)")
    
    print("Creating and distributing training dataset...")
    training_tf_dataset = create_dataset_from_hf_stream(
        hf_dataset_name_str=HF_DATASET_NAME, 
        hf_split_name_str=HF_DATASET_SPLIT, 
        tokenizer_instance_obj=tokenizer, 
        global_tf_data_batch_size_val=global_batch_size_val,
        max_sequence_length_val=MAX_SEQ_LEN, 
        num_mtp_heads_val=NUM_MTP_HEADS, 
        eos_token_id_val=EOS_TOKEN_ID
    )
    distributed_training_tf_dataset = distributed_strategy_instance.experimental_distribute_dataset(training_tf_dataset)
    print("Dataset pipeline created and distributed successfully.")
    
    with distributed_strategy_instance.scope():
        print("Building Keras model under strategy scope...")
        keras_model_main_instance = ExpG3_build_model(eject_mtp_path_flag=(NUM_MTP_HEADS == 0))
        keras_model_main_instance.summary(line_length=130)
        
        num_decay_steps_for_lr = max(1, current_total_train_steps - WARMUP_STEPS)
        learning_rate_schedule_obj = CustomLearningRateSchedule(
            peak_learning_rate_val=LEARNING_RATE, 
            warmup_steps_val=WARMUP_STEPS, 
            total_decay_steps_val=num_decay_steps_for_lr
        )
        
        adamw_optimizer_obj = keras.optimizers.AdamW(
            learning_rate=learning_rate_schedule_obj,
            weight_decay=LEARNING_RATE * 0.01, # Example weight decay
            beta_1=0.9, beta_2=0.98, epsilon=1e-6,
            clipnorm=CLIP_NORM 
        )
        
        sparse_categorical_loss_function_obj = keras.losses.SparseCategoricalCrossentropy(
            from_logits=False, # Model outputs probabilities due to Softmax in OutputHead
            reduction="none" # We need per-example loss for manual averaging
        )
        
        def calculate_loss_for_replica_batch(list_of_labels_per_head, list_of_predictions_per_head):
            sum_of_weighted_losses_on_replica = 0.0
            num_prediction_heads_val = len(list_of_predictions_per_head)
            # Simple equal weighting for now, can be customized
            loss_weights_for_each_head = [1.0] * num_prediction_heads_val 
            
            batch_size_on_current_replica = tf.cast(tf.shape(list_of_labels_per_head[0])[0], tf.float32)
            if batch_size_on_current_replica == 0.0: 
                return 0.0 # Avoid division by zero

            for head_index_val in range(num_prediction_heads_val):
                # In this setup, each head predicts the *next* token after its input sequence.
                # The input sequence length to the model is MAX_SEQ_LEN.
                # The output sequence length from G3Block is MAX_SEQ_LEN / 2^(NUM_MODEL_LAYERS)
                # The output sequence length from MTPG3Block is same as its input (passed from previous block)
                # The labels correspond to the token that *would have followed* the sequence processed by that head.
                # The predictions are [Batch, Seq_Len_for_Head, Vocab]. We need the prediction for the *last* token in that head's output sequence.
                
                # The labels `list_of_labels_per_head[head_index_val]` are shaped (Batch, 1)
                # The predictions `list_of_predictions_per_head[head_index_val]` are (Batch, Head_Seq_Len, Vocab)
                # We take the logits from the last time step of the head's output sequence.
                predictions_at_last_time_step = list_of_predictions_per_head[head_index_val][:, -1, :] # (Batch, Vocab)
                
                loss_for_current_head_on_batch = sparse_categorical_loss_function_obj(
                    list_of_labels_per_head[head_index_val], # (Batch, 1)
                    predictions_at_last_time_step # (Batch, Vocab)
                ) # Output shape (Batch,)
                sum_of_weighted_losses_on_replica += loss_weights_for_each_head[head_index_val] * tf.reduce_sum(loss_for_current_head_on_batch)
            
            total_sum_of_loss_weights = sum(loss_weights_for_each_head)
            if total_sum_of_loss_weights == 0.0: 
                return 0.0
            
            # Average loss per example *across all heads on this replica*
            average_loss_per_example_replica = sum_of_weighted_losses_on_replica / (batch_size_on_current_replica * total_sum_of_loss_weights)
            return average_loss_per_example_replica
            
        training_loss_metric_tracker = keras.metrics.Mean(name='training_loss_metric_tracker')
        print("Keras model, optimizer, loss function, and metric initialized successfully under strategy scope.")

    @tf.function
    def distributed_training_step_function(distributed_model_inputs_tuple):
        (batch_input_ids_tensor, batch_initial_memory_tensor), batch_target_labels_list_tuple = distributed_model_inputs_tuple
        
        with tf.GradientTape() as KerasGradientTape: # Use a more descriptive name
            batch_predictions_list_all_heads, _ = keras_model_main_instance(
                [batch_input_ids_tensor, batch_initial_memory_tensor], training=True
            )
            loss_per_example_this_replica = calculate_loss_for_replica_batch(batch_target_labels_list_tuple, batch_predictions_list_all_heads)
            
            loss_for_gradient_calculation = loss_per_example_this_replica
            # Keras 3 AdamW optimizer with TF backend should handle scaling automatically when policy is set
            # No explicit get_scaled_loss / get_unscaled_gradients needed usually.
        
        gradients_for_trainable_vars = KerasGradientTape.gradient(loss_for_gradient_calculation, keras_model_main_instance.trainable_variables)
        
        adamw_optimizer_obj.apply_gradients(zip(gradients_for_trainable_vars, keras_model_main_instance.trainable_variables))
        training_loss_metric_tracker.update_state(loss_per_example_this_replica)
        return loss_per_example_this_replica

    print(f"Starting Training (Target Global Steps: {current_total_train_steps})...")
    for epoch_index_val in range(EPOCHS):
        print(f"\nEpoch {epoch_index_val + 1}/{EPOCHS}")
        
        distributed_dataset_epoch_iterator = iter(distributed_training_tf_dataset)
        global_step_count_at_epoch_start = adamw_optimizer_obj.iterations.numpy()
        
        if global_step_count_at_epoch_start >= current_total_train_steps:
            print("Target total training steps already reached before starting this epoch. Stopping.")
            break
        
        num_steps_to_run_this_epoch = current_total_train_steps - global_step_count_at_epoch_start
        if EPOCHS > 1 and epoch_index_val < EPOCHS - 1: # If multi-epoch, try to balance steps per epoch
            nominal_steps_per_epoch_val = (current_total_train_steps // EPOCHS) if EPOCHS > 0 else current_total_train_steps
            num_steps_to_run_this_epoch = min(num_steps_to_run_this_epoch, nominal_steps_per_epoch_val)
        num_steps_to_run_this_epoch = max(0, int(num_steps_to_run_this_epoch)) # Ensure non-negative integer
        
        if num_steps_to_run_this_epoch == 0 and global_step_count_at_epoch_start < current_total_train_steps : # If rounding made it 0 but still steps left
             num_steps_to_run_this_epoch = 1 
        if num_steps_to_run_this_epoch == 0: break # No more steps to run

        epoch_progress_bar_display = tqdm.tqdm(
            range(num_steps_to_run_this_epoch), 
            desc=f"Epoch {epoch_index_val + 1}", 
            unit="step", 
            leave=True
        )
        
        for _ in epoch_progress_bar_display:
            current_global_step_count_iter = adamw_optimizer_obj.iterations.numpy()
            if current_global_step_count_iter >= current_total_train_steps: break # Check before running step
            
            try:
                # Run the training step on the distributed dataset
                distributed_strategy_instance.run(distributed_training_step_function, args=(next(distributed_dataset_epoch_iterator),))
            except tf.errors.OutOfRangeError:
                print("Dataset iterator exhausted for this epoch.")
                break # Exit step loop for this epoch
            except Exception as error_during_step_execution:
                print(f"Error during training at Global Step {current_global_step_count_iter}: {error_during_step_execution}")
                print(traceback.format_exc())
                # Potentially save model or state here before exiting
                return # Exit main function on critical error
            
            epoch_progress_bar_display.set_postfix({
                "Loss": f"{training_loss_metric_tracker.result():.4f}",
                "LR": f"{adamw_optimizer_obj.learning_rate(current_global_step_count_iter):.2e}", # Access LR from optimizer
                "GS": f"{current_global_step_count_iter + 1}" # +1 because iterations updates after apply_gradients
            })
            
            if (current_global_step_count_iter + 1) % 100 == 0 and (current_global_step_count_iter + 1) < current_total_train_steps:
                print(f"Epoch {epoch_index_val+1}, Global Step {current_global_step_count_iter+1}: "
                      f"Loss {training_loss_metric_tracker.result():.4f}, "
                      f"LR {adamw_optimizer_obj.learning_rate(current_global_step_count_iter):.2e}")
        
        print(f"Epoch {epoch_index_val+1} finished. Global Step: {adamw_optimizer_obj.iterations.numpy()}. "
              f"Average Loss for Epoch: {training_loss_metric_tracker.result():.4f}")
        training_loss_metric_tracker.reset_states()
        
        if adamw_optimizer_obj.iterations.numpy() >= current_total_train_steps:
            print("Target total training steps reached. Exiting training loop.")
            break
            
    print("\nTraining Loop Finished.")

    model_weights_filepath_str = "./exp_g3_model_final_weights.weights.h5" 
    if adamw_optimizer_obj.iterations.numpy() >= current_total_train_steps and current_total_train_steps > 0:
        print(f"Saving final model weights to {model_weights_filepath_str}...")
        try:
            keras_model_main_instance.save_weights(model_weights_filepath_str)
            print("Model weights saved successfully.")
        except Exception as error_saving_model_weights:
            print(f"Error saving model weights: {error_saving_model_weights}\n{traceback.format_exc()}")
    else:
        print(f"Skipping model weights saving: training did not complete target steps "
              f"({adamw_optimizer_obj.iterations.numpy()}/{current_total_train_steps} steps).")

    # Perform text generation only if training completed sufficiently
    if adamw_optimizer_obj.iterations.numpy() >= current_total_train_steps and current_total_train_steps > 0:
        print("\n--- Text Generation Test Post-Training ---")
        initial_memory_for_generation_numpy_array = np.zeros((NUM_REGISTERS, MEMORY_DIM), dtype=np.float32)
        test_prompts_list_for_generation = [
            "The future of Artificial Intelligence is looking", 
            "TensorFlow and Keras are frameworks for", 
            "My house cat often stares at the wall, probably contemplating", 
            "Advanced machine learning models can be used to"
        ]
        for current_test_prompt_string in test_prompts_list_for_generation:
            try:
                generate_text_from_model(
                    tpu_strategy_arg=distributed_strategy_instance, # Passed but not used inside for model call
                    keras_model_to_generate_from=keras_model_main_instance, 
                    tokenizer_to_use=tokenizer, 
                    input_prompt_string=current_test_prompt_string, 
                    initial_memory_state_numpy=initial_memory_for_generation_numpy_array.copy(), # Use a copy
                    max_tokens_to_generate=70, 
                    text_sampling_temperature=0.75
                )
            except Exception as error_during_text_generation:
                print(f"Error during text generation for prompt '{current_test_prompt_string}': {error_during_text_generation}\n{traceback.format_exc()}")
            print("-" * 50) 
        print("\n--- Text Generation Test Finished ---")
    else:
        print("\nSkipping Text Generation: Training was incomplete or too short.")

if __name__ == '__main__':
    cleanup_tensorflow_runtimes() 
    # tokenizer global is initialized by initialize_tokenizer() if it's None within main()
    main(total_train_steps_override=TOTAL_TRAIN_STEPS_CONFIG)
