#  -*- coding: utf-8 -*-
#
#  Copyright 2019 Jim Martens
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
Functionality to run my auto-encoder implementation.

This module provides a function to run a trained simple auto-encoder.

Functions:
    run_simple(...): runs a trained simple auto-encoder
"""
import os
import time
from typing import Dict, Tuple

import tensorflow as tf
from tensorflow.python.ops import summary_ops_v2

# shortcuts for tensorflow sub packages and classes
from twomartens.masterthesis.aae import model, train, util

K = tf.keras.backend
tfe = tf.contrib.eager


def run_simple(dataset: tf.data.Dataset,
               iteration: int,
               weights_prefix: str,
               channels: int = 3,
               zsize: int = 64,
               batch_size: int = 16,
               verbose: bool = False) -> None:
    """
    Runs the trained auto-encoder for given data set.

    This function runs the trained auto-encoder

    Args:
        dataset: run dataset
        iteration: identifier for the used training run
        weights_prefix: prefix for trained weights directory
        channels: number of channels in input image (default: 3)
        zsize: size of the intermediary z (default: 64)
        batch_size: size of each batch (default: 16)
        verbose: if True training progress is printed to console (default: False)
    """
    
    # checkpointed tensors and variables
    checkpointables = {
        # get models
        'encoder':             model.Encoder(zsize),
        'decoder':             model.Decoder(channels, zsize),
    }
    
    global_step = tf.train.get_or_create_global_step()
    
    # checkpoint
    checkpoint_dir = os.path.join(weights_prefix, str(iteration) + '/')
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    checkpoint = tf.train.Checkpoint(**checkpointables)
    checkpoint.restore(latest_checkpoint)
    
    outputs = _run_one_epoch_simple(dataset,
                                    batch_size=batch_size,
                                    global_step=global_step,
                                    **checkpointables)
    
    if verbose:
        print((
            f"run time: {outputs['time']:.2f}, "
            f"Encoder + Decoder loss: {outputs['enc_dec_loss']:.3f}"
        ))


def _run_one_epoch_simple(dataset: tf.data.Dataset,
                          batch_size: int,
                          encoder: model.Encoder,
                          decoder: model.Decoder,
                          global_step: tf.Variable) -> Dict[str, float]:
    with summary_ops_v2.always_record_summaries():
        start_time = time.time()
        enc_dec_loss_avg = tfe.metrics.Mean(name='encoder_decoder_loss',
                                            dtype=tf.float32)
        
        for x in dataset:
            reconstruction_loss, x_decoded = _run_enc_dec_step_simple(encoder=encoder,
                                                                      decoder=decoder,
                                                                      inputs=x,
                                                                      global_step=global_step)
            enc_dec_loss_avg(reconstruction_loss)
            
            if int(global_step % train.LOG_FREQUENCY) == 0:
                comparison = K.concatenate([x[:int(batch_size / 2)], x_decoded[:int(batch_size / 2)]], axis=0)
                grid = util.prepare_image(comparison.cpu(), nrow=int(batch_size / 2))
                summary_ops_v2.image(name='reconstruction',
                                     tensor=K.expand_dims(grid, axis=0), max_images=1,
                                     step=global_step)
            global_step.assign_add(1)
        
        end_time = time.time()
        run_time = end_time - start_time
        
        # final losses of epoch
        outputs = {
            'enc_dec_loss': enc_dec_loss_avg.result(False),
            'run_time': run_time
        }
        
        return outputs


def _run_enc_dec_step_simple(encoder: model.Encoder, decoder: model.Decoder,
                             inputs: tf.Tensor,
                             global_step: tf.Variable) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Runs the encoder and decoder jointly for one step (one batch).
    
    Args:
        encoder: instance of encoder model
        decoder: instance of decoder model
        inputs: inputs from data set
        global_step: the global step variable

    Returns:
        tuple of reconstruction loss, reconstructed input, latent space value
    """
    z = encoder(inputs)
    x_decoded = decoder(z)
    
    reconstruction_loss = tf.losses.log_loss(inputs, x_decoded)
    
    if int(global_step % train.LOG_FREQUENCY) == 0:
        summary_ops_v2.scalar(name='reconstruction_loss', tensor=reconstruction_loss,
                              step=global_step)
    
    return reconstruction_loss, x_decoded
