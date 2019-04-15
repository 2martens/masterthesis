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
Training functionality for my AAE implementation.

This module provides a function to train a simple auto-encoder.

Attributes:
    LOG_FREQUENCY: number of steps that must pass before logging happens

Functions:
    train_simple(...): trains a simple auto-encoder only with reconstruction loss

"""
import os
import time
from typing import Dict
from typing import Tuple

import tensorflow as tf
from tensorflow.python.ops import summary_ops_v2

from twomartens.masterthesis.aae import model
from twomartens.masterthesis.aae import util

# shortcuts for tensorflow sub packages and classes
K = tf.keras.backend
tfe = tf.contrib.eager

LOG_FREQUENCY: int = 10


def train_simple(dataset: tf.data.Dataset,
                 iteration: int,
                 weights_prefix: str,
                 channels: int = 1,
                 zsize: int = 32,
                 lr: float = 0.002,
                 train_epoch: int = 80,
                 batch_size: int = 128,
                 verbose: bool = True) -> None:
    """
    Trains auto-encoder for given data set.

    This function creates checkpoints after every
    epoch as well as after finishing training (or stopping early). When starting
    this function with the same ``iteration`` then the training will try to
    continue where it ended last time by restoring a saved checkpoint.
    The loss values are provided as scalar summaries. Reconstruction images are
    provided as summary images.

    Args:
        dataset: train dataset
        iteration: identifier for the current training run
        weights_prefix: prefix for weights directory
        channels: number of channels in input image (default: 1)
        zsize: size of the intermediary z (default: 32)
        lr: initial learning rate (default: 0.002)
        train_epoch: number of epochs to train (default: 80)
        batch_size: size of each batch (default: 128)
        verbose: if True prints train progress info to console (default: True)
    """
    
    # checkpointed tensors and variables
    checkpointables = {
        'learning_rate_var': K.variable(lr),
    }
    checkpointables.update({
        # get models
        'encoder':             model.Encoder(zsize),
        'decoder':             model.Decoder(channels, zsize),
        # define optimizers
        'enc_dec_optimizer':   tf.train.AdamOptimizer(learning_rate=checkpointables['learning_rate_var']),
        # global step counter
        'epoch_var':           K.variable(-1, dtype=tf.int64),
        'global_step':         tf.train.get_or_create_global_step(),
        'global_step_enc_dec': K.variable(0, dtype=tf.int64),
    })
    
    # checkpoint
    checkpoint_dir = os.path.join(weights_prefix, str(iteration) + '/')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    checkpoint = tf.train.Checkpoint(**checkpointables)
    checkpoint.restore(latest_checkpoint)
    
    def _get_last_epoch(epoch_var: tf.Variable, **kwargs) -> int:
        return int(epoch_var)
    
    last_epoch = _get_last_epoch(**checkpointables)
    previous_epochs = 0
    if last_epoch != -1:
        previous_epochs = last_epoch + 1
    
    with summary_ops_v2.always_record_summaries():
        summary_ops_v2.scalar(name='learning_rate', tensor=checkpointables['learning_rate_var'],
                              step=checkpointables['global_step'])
    
    for epoch in range(train_epoch - previous_epochs):
        _epoch = epoch + previous_epochs
        outputs = _train_one_epoch_simple(_epoch, dataset,
                                          verbose=verbose,
                                          batch_size=batch_size,
                                          **checkpointables)
        
        if verbose:
            print((
                f"[{_epoch + 1:d}/{train_epoch:d}] - "
                f"train time: {outputs['per_epoch_time']:.2f}, "
                f"Encoder + Decoder loss: {outputs['enc_dec_loss']:.3f}"
            ))
        
        # save weights at end of epoch
        checkpoint.save(checkpoint_prefix)
    
    if verbose:
        print("Training finish!... save model weights")
    
    # save trained models
    checkpoint.save(checkpoint_prefix)


def _train_one_epoch_simple(epoch: int,
                            dataset: tf.data.Dataset,
                            verbose: bool,
                            batch_size: int,
                            learning_rate_var: tf.Variable,
                            decoder: model.Decoder,
                            encoder: model.Encoder,
                            enc_dec_optimizer: tf.train.Optimizer,
                            global_step: tf.Variable,
                            global_step_enc_dec: tf.Variable,
                            epoch_var: tf.Variable) -> Dict[str, float]:
    with summary_ops_v2.always_record_summaries():
        epoch_var.assign(epoch)
        epoch_start_time = time.time()
        # define loss variables
        enc_dec_loss_avg = tfe.metrics.Mean(name='encoder_decoder_loss', dtype=tf.float32)
        
        # update learning rate
        if (epoch + 1) % 30 == 0:
            learning_rate_var.assign(learning_rate_var.value() / 4)
            summary_ops_v2.scalar(name='learning_rate', tensor=learning_rate_var,
                                  step=global_step)
            if verbose:
                print("learning rate change!")

        for x in dataset:
            reconstruction_loss, x_decoded = _train_enc_dec_step_simple(encoder=encoder,
                                                                        decoder=decoder,
                                                                        optimizer=enc_dec_optimizer,
                                                                        inputs=x,
                                                                        global_step_enc_dec=global_step_enc_dec,
                                                                        global_step=global_step)
            enc_dec_loss_avg(reconstruction_loss)
            
            if int(global_step % LOG_FREQUENCY) == 0:
                comparison = K.concatenate([x[:int(batch_size / 2)], x_decoded[:int(batch_size / 2)]], axis=0)
                grid = util.prepare_image(comparison.cpu(), nrow=int(batch_size/2))
                summary_ops_v2.image(name='reconstruction',
                                     tensor=K.expand_dims(grid, axis=0), max_images=1,
                                     step=global_step)
            global_step.assign_add(1)
        
        epoch_end_time = time.time()
        per_epoch_time = epoch_end_time - epoch_start_time
        
        # final losses of epoch
        outputs = {
            'enc_dec_loss':   enc_dec_loss_avg.result(False),
            'per_epoch_time': per_epoch_time,
        }
        
        return outputs


def _train_enc_dec_step_simple(encoder: model.Encoder, decoder: model.Decoder,
                               optimizer: tf.train.Optimizer,
                               inputs: tf.Tensor,
                               global_step: tf.Variable,
                               global_step_enc_dec: tf.Variable) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Trains the encoder and decoder jointly for one step (one batch).

    :param encoder: instance of encoder model
    :param decoder: instance of decoder model
    :param optimizer: instance of chosen optimizer
    :param inputs: inputs from data set
    :param global_step: the global step variable
    :param global_step_enc_dec: global step variable for enc_dec
    :return: tuple of reconstruction loss, reconstructed input
    """
    with tf.GradientTape() as tape:
        z = encoder(inputs)
        x_decoded = decoder(z)
        
        reconstruction_loss = tf.losses.log_loss(inputs, x_decoded)
    
    enc_dec_grads = tape.gradient(reconstruction_loss,
                                  encoder.trainable_variables + decoder.trainable_variables)
    if int(global_step % LOG_FREQUENCY) == 0:
        summary_ops_v2.scalar(name='reconstruction_loss', tensor=reconstruction_loss,
                              step=global_step)
        for grad, variable in zip(enc_dec_grads, encoder.trainable_variables + decoder.trainable_variables):
            summary_ops_v2.histogram(name='gradients/' + variable.name, tensor=tf.math.l2_normalize(grad),
                                     step=global_step)
            summary_ops_v2.histogram(name='variables/' + variable.name, tensor=tf.math.l2_normalize(variable),
                                     step=global_step)
    optimizer.apply_gradients(zip(enc_dec_grads,
                                  encoder.trainable_variables + decoder.trainable_variables),
                              global_step=global_step_enc_dec)
    
    return reconstruction_loss, x_decoded


if __name__ == "__main__":
    from twomartens.masterthesis.aae.data import prepare_training_data
    tf.enable_eager_execution()
    inlier_classes = [8]
    iteration = 2
    train_dataset, _ = prepare_training_data(test_fold_id=0, inlier_classes=inlier_classes,
                                             total_classes=10)
    train_summary_writer = summary_ops_v2.create_file_writer(
        './summaries/train/number-' + str(inlier_classes[0]) + '/' + str(iteration))
    with train_summary_writer.as_default():
        train_simple(dataset=train_dataset, iteration=iteration,
                     weights_prefix='weights/' + str(inlier_classes[0]) + '/')
