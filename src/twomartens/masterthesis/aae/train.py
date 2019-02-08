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

This module provides functions to prepare the training data and subsequently
train the Adversarial Auto Encoder.

Attributes:
    GRACE: specifies the number of epochs that the training loss can stagnate or worsen
        before the training is stopped early
    TOTAL_LOSS_GRACE_CAP: upper limit for total loss, grace countdown only enabled if total loss higher
    LOG_FREQUENCY: number of steps that must pass before logging happens

Functions:
    prepare_training_data(...): prepares the mnist training data
    train(...): trains the AAE models

Todos:
    - fix early stopping
    - fix losses reaching exactly zero

"""

import functools
import math
import os
import pickle
import time
from typing import Callable
from typing import Dict
from typing import Sequence
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import summary_ops_v2

from twomartens.masterthesis.aae import model
from twomartens.masterthesis.aae import util

# shortcuts for tensorflow sub packages and classes
K = tf.keras.backend
tfe = tf.contrib.eager

GRACE: int = 10
TOTAL_LOSS_GRACE_CAP: int = 6
LOG_FREQUENCY: int = 10


def prepare_training_data(test_fold_id: int,
                          inlier_classes: Sequence[int],
                          total_classes: int,
                          fold_prefix: str = 'data/data_fold_',
                          batch_size: int = 128,
                          folds: int = 5) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Prepares the MNIST training data.
    
    Args:
        test_fold_id: id of test fold
        inlier_classes: list of class ids that are considered inliers
        total_classes: total number of classes
        fold_prefix: the prefix for the fold pickle files (default: 'data/data_fold_')
        batch_size: size of batch (default: 128)
        folds: number of folds (default: 5)
    
    Returns:
        A tuple (train dataset, valid dataset)
    """
    # prepare data
    mnist_train = []
    mnist_valid = []
    
    for i in range(folds):
        if i != test_fold_id:  # exclude testing fold, representing 20% of each class
            with open(f"{fold_prefix}{i:d}.pkl", 'rb') as pkl:
                fold = pickle.load(pkl)
            if len(mnist_valid) == 0:  # single out one fold, comprising 20% of each class
                mnist_valid = fold
            else:  # form train set from remaining folds, comprising 60% of each class
                mnist_train += fold
    
    outlier_classes = []
    for i in range(total_classes):
        if i not in inlier_classes:
            outlier_classes.append(i)
    
    # keep only train classes
    mnist_train = [x for x in mnist_train if x[0] in inlier_classes]
    
    def _list_of_pairs_to_numpy(list_of_pairs: Sequence[Tuple[int, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts a list of pairs to a numpy array.

        Args:
            list_of_pairs: list of pairs
        
        Returns:
            tuple (feature array, label array)
        """
        return np.asarray([x[1] for x in list_of_pairs], np.float32), np.asarray([x[0] for x in list_of_pairs], np.int)
    
    mnist_train_x, mnist_train_y = _list_of_pairs_to_numpy(mnist_train)
    mnist_valid_x, mnist_valid_y = _list_of_pairs_to_numpy(mnist_valid)
    
    # get dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((mnist_train_x, mnist_train_y))
    train_dataset = train_dataset.shuffle(mnist_train_x.shape[0]).batch(batch_size,
                                                                        drop_remainder=True).map(_normalize)
    valid_dataset = tf.data.Dataset.from_tensor_slices((mnist_valid_x, mnist_valid_y))
    valid_dataset = valid_dataset.shuffle(mnist_valid_x.shape[0]).batch(batch_size,
                                                                        drop_remainder=True).map(_normalize)
    
    return train_dataset, valid_dataset


def train(dataset: tf.data.Dataset,
          iteration: int,
          weights_prefix: str,
          channels: int = 1,
          zsize: int = 32,
          lr: float = 0.002,
          batch_size: int = 128,
          train_epoch: int = 80,
          verbose: bool = True,
          early_stopping: bool = False) -> None:
    """
    Trains AAE for given data set.
    
    This function provides early stopping and creates checkpoints after every
    epoch as well as after finishing training (or stopping early). When starting
    this function with the same ``iteration`` then the training will try to
    continue where it ended last time by restoring a saved checkpoint.
    The loss values are provided as scalar summaries. Reconstruction and sample
    images are provided as summary images.
    
    Args:
        dataset: train dataset
        iteration: identifier for the current training run
        weights_prefix: prefix for weights directory
        channels: number of channels in input image (default: 1)
        zsize: size of the intermediary z (default: 32)
        lr: initial learning rate (default: 0.002)
        batch_size: the size of each batch (default: 128)
        train_epoch: number of epochs to train (default: 80)
        verbose: if True prints train progress info to console (default: True)
        early_stopping: if True the early stopping mechanic is enabled (default: False)
        
    Notes:
        The training stops early if for ``GRACE`` number of epochs the loss is not
        decreasing. Specifically all individual losses are accounted for and any one
        of those not decreasing triggers a ``strike``. If the total loss, which is
        a sum of all individual losses, is also not decreasing and has a total
        value of more than ``TOTAL_LOSS_GRACE_CAP``, the counter for the remaining grace period is
        decreased. If in any epoch afterwards all losses are decreasing the grace
        period is reset to ``GRACE``. Lastly the training loop will be stopped early
        if the grace counter reaches ``0`` at the end of an epoch.
    """
    
    # non-preserved tensors
    y_real = K.ones(batch_size)
    y_fake = K.zeros(batch_size)
    sample = K.expand_dims(K.expand_dims(K.random_normal((64, zsize)), axis=1), axis=1)
    # z generator function
    z_generator = functools.partial(_get_z_variable, batch_size=batch_size, zsize=zsize)

    # non-preserved python variables
    encoder_lowest_loss = math.inf
    decoder_lowest_loss = math.inf
    enc_dec_lowest_loss = math.inf
    zd_lowest_loss = math.inf
    xd_lowest_loss = math.inf
    total_lowest_loss = math.inf
    grace_period = GRACE
    
    # checkpointed tensors and variables
    checkpointables = {
        'learning_rate_var': K.variable(lr),
    }
    checkpointables.update({
        # get models
        'encoder': model.Encoder(zsize),
        'decoder': model.Decoder(channels),
        'z_discriminator': model.ZDiscriminator(),
        'x_discriminator': model.XDiscriminator(),
        # define optimizers
        'decoder_optimizer': tf.train.AdamOptimizer(learning_rate=checkpointables['learning_rate_var'], beta1=0.5, beta2=0.999),
        'enc_dec_optimizer': tf.train.AdamOptimizer(learning_rate=checkpointables['learning_rate_var'], beta1=0.5, beta2=0.999),
        'z_discriminator_optimizer': tf.train.AdamOptimizer(learning_rate=checkpointables['learning_rate_var'],
                                                            beta1=0.5, beta2=0.999),
        'x_discriminator_optimizer': tf.train.AdamOptimizer(learning_rate=checkpointables['learning_rate_var'],
                                                            beta1=0.5, beta2=0.999),
        # global step counter
        'epoch_var': K.variable(-1, dtype=tf.int64),
        'global_step': tf.train.get_or_create_global_step(),
        'global_step_decoder': K.variable(0, dtype=tf.int64),
        'global_step_enc_dec': K.variable(0, dtype=tf.int64),
        'global_step_xd': K.variable(0, dtype=tf.int64),
        'global_step_zd': K.variable(0, dtype=tf.int64),
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
        outputs = _train_one_epoch(_epoch, dataset, targets_real=y_real,
                                   targets_fake=y_fake, z_generator=z_generator,
                                   verbose=verbose,
                                   **checkpointables)
        
        if verbose:
            print((
                f"[{_epoch + 1:d}/{train_epoch:d}] - "
                f"train time: {outputs['per_epoch_time']:.2f}, "
                f"Decoder loss: {outputs['decoder_loss']:.3f}, "
                f"X Discriminator loss: {outputs['xd_loss']:.3f}, "
                f"Z Discriminator loss: {outputs['zd_loss']:.3f}, "
                f"Encoder + Decoder loss: {outputs['enc_dec_loss']:.3f}, "
                f"Encoder loss: {outputs['encoder_loss']:.3f}"
            ))
        
        # save sample image summary
        def _save_sample(decoder: model.Decoder, global_step: tf.Variable, **kwargs) -> None:
            resultsample = decoder(sample).cpu()
            grid = util.prepare_image(resultsample)
            summary_ops_v2.image(name='sample', tensor=K.expand_dims(grid, axis=0),
                                 max_images=1, step=global_step)
        
        with summary_ops_v2.always_record_summaries():
            _save_sample(**checkpointables)
        
        # save weights at end of epoch
        checkpoint.save(checkpoint_prefix)

        # check for improvements in error reduction - otherwise early stopping
        if early_stopping:
            strike = False
            total_strike = False
            total_loss = outputs['encoder_loss'] + outputs['decoder_loss'] + outputs['enc_dec_loss'] + \
                outputs['xd_loss'] + outputs['zd_loss']
            if total_loss < total_lowest_loss:
                total_lowest_loss = total_loss
            elif total_loss > TOTAL_LOSS_GRACE_CAP:
                total_strike = True
            if outputs['encoder_loss'] < encoder_lowest_loss:
                encoder_lowest_loss = outputs['encoder_loss']
            else:
                strike = True
            if outputs['decoder_loss'] < decoder_lowest_loss:
                decoder_lowest_loss = outputs['decoder_loss']
            else:
                strike = True
            if outputs['enc_dec_loss'] < enc_dec_lowest_loss:
                enc_dec_lowest_loss = outputs['enc_dec_loss']
            else:
                strike = True
            if outputs['xd_loss'] < xd_lowest_loss:
                xd_lowest_loss = outputs['xd_loss']
            else:
                strike = True
            if outputs['zd_loss'] < zd_lowest_loss:
                zd_lowest_loss = outputs['zd_loss']
            else:
                strike = True
            
            if strike and total_strike:
                grace_period -= 1
            elif strike:
                pass
            else:
                grace_period = GRACE
                
            if grace_period == 0:
                break
        
    if verbose:
        if grace_period > 0:
            print("Training finish!... save model weights")
        if grace_period == 0:
            print("Training stopped early!... save model weights")

    # save trained models
    checkpoint.save(checkpoint_prefix)


def _train_one_epoch(epoch: int,
                     dataset: tf.data.Dataset,
                     targets_real: tf.Tensor,
                     verbose: bool,
                     targets_fake: tf.Tensor,
                     z_generator: Callable[[], tf.Variable],
                     learning_rate_var: tf.Variable,
                     decoder: model.Decoder,
                     encoder: model.Encoder,
                     x_discriminator: model.XDiscriminator,
                     z_discriminator: model.ZDiscriminator,
                     decoder_optimizer: tf.train.Optimizer,
                     x_discriminator_optimizer: tf.train.Optimizer,
                     z_discriminator_optimizer: tf.train.Optimizer,
                     enc_dec_optimizer: tf.train.Optimizer,
                     global_step: tf.Variable,
                     global_step_xd: tf.Variable,
                     global_step_zd: tf.Variable,
                     global_step_decoder: tf.Variable,
                     global_step_enc_dec: tf.Variable,
                     epoch_var: tf.Variable) -> Dict[str, float]:
    
    with summary_ops_v2.always_record_summaries():
        epoch_var.assign(epoch)
        epoch_start_time = time.time()
        # define loss variables
        encoder_loss_avg = tfe.metrics.Mean(name='encoder_loss', dtype=tf.float32)
        decoder_loss_avg = tfe.metrics.Mean(name='decoder_loss', dtype=tf.float32)
        enc_dec_loss_avg = tfe.metrics.Mean(name='encoder_decoder_loss', dtype=tf.float32)
        zd_loss_avg = tfe.metrics.Mean(name='z_discriminator_loss', dtype=tf.float32)
        xd_loss_avg = tfe.metrics.Mean(name='x_discriminator_loss', dtype=tf.float32)
        
        # update learning rate
        if (epoch + 1) % 30 == 0:
            learning_rate_var.assign(learning_rate_var.value() / 4)
            summary_ops_v2.scalar(name='learning_rate', tensor=learning_rate_var,
                                  step=global_step)
            if verbose:
                print("learning rate change!")
        
        for x, _ in dataset:
            # x discriminator
            _xd_train_loss = _train_xdiscriminator_step(x_discriminator=x_discriminator,
                                                        decoder=decoder,
                                                        optimizer=x_discriminator_optimizer,
                                                        inputs=x,
                                                        targets_real=targets_real,
                                                        targets_fake=targets_fake,
                                                        global_step_xd=global_step_xd,
                                                        global_step=global_step,
                                                        z_generator=z_generator)
            xd_loss_avg(_xd_train_loss)
            
            # --------
            # decoder
            _decoder_train_loss = _train_decoder_step(decoder=decoder,
                                                      x_discriminator=x_discriminator,
                                                      optimizer=decoder_optimizer,
                                                      targets=targets_real,
                                                      global_step_decoder=global_step_decoder,
                                                      global_step=global_step,
                                                      z_generator=z_generator)
            decoder_loss_avg(_decoder_train_loss)
            
            # ---------
            # z discriminator
            _zd_train_loss = _train_zdiscriminator_step(z_discriminator=z_discriminator,
                                                        encoder=encoder,
                                                        optimizer=z_discriminator_optimizer,
                                                        inputs=x,
                                                        targets_real=targets_real,
                                                        targets_fake=targets_fake,
                                                        global_step_zd=global_step_zd,
                                                        global_step=global_step,
                                                        z_generator=z_generator)
            zd_loss_avg(_zd_train_loss)
            
            # -----------
            # encoder + decoder
            encoder_loss, reconstruction_loss, x_decoded = _train_enc_dec_step(encoder=encoder,
                                                                               decoder=decoder,
                                                                               z_discriminator=z_discriminator,
                                                                               optimizer=enc_dec_optimizer,
                                                                               inputs=x,
                                                                               targets=targets_real,
                                                                               global_step_enc_dec=global_step_enc_dec,
                                                                               global_step=global_step)
            enc_dec_loss_avg(reconstruction_loss)
            encoder_loss_avg(encoder_loss)
            
            if int(global_step % LOG_FREQUENCY) == 0:
                comparison = K.concatenate([x[:64], x_decoded[:64]], axis=0)
                grid = util.prepare_image(comparison.cpu(), nrow=64)
                summary_ops_v2.image(name='reconstruction',
                                     tensor=K.expand_dims(grid, axis=0), max_images=1,
                                     step=global_step)
            global_step.assign_add(1)
        
        epoch_end_time = time.time()
        per_epoch_time = epoch_end_time - epoch_start_time
        
        # final losses of epoch
        outputs = {
            'decoder_loss': decoder_loss_avg.result(False),
            'encoder_loss': encoder_loss_avg.result(False),
            'enc_dec_loss': enc_dec_loss_avg.result(False),
            'xd_loss': xd_loss_avg.result(False),
            'zd_loss': zd_loss_avg.result(False),
            'per_epoch_time': per_epoch_time,
        }
        
        return outputs


def _train_xdiscriminator_step(x_discriminator: model.XDiscriminator,
                               decoder: model.Decoder,
                               optimizer: tf.train.Optimizer,
                               inputs: tf.Tensor,
                               targets_real: tf.Tensor,
                               targets_fake: tf.Tensor,
                               global_step: tf.Variable,
                               global_step_xd: tf.Variable,
                               z_generator: Callable[[], tf.Variable]) -> tf.Tensor:
    """
    Trains the x discriminator model for one step (one batch).
    
    :param x_discriminator: instance of x discriminator model
    :param decoder: instance of decoder model
    :param optimizer: instance of chosen optimizer
    :param inputs: inputs from dataset
    :param targets_real: target tensor for real loss calculation
    :param targets_fake: target tensor for fake loss calculation
    :param global_step: the global step variable
    :param global_step_xd: global step variable for xd
    :param z_generator: callable function that returns a z variable
    :return: the calculated loss
    """
    with tf.GradientTape() as tape:
        xd_result_1 = tf.squeeze(x_discriminator(inputs))
        xd_real_loss = tf.losses.log_loss(targets_real, xd_result_1)
        
        z = z_generator()
        x_fake = decoder(z)
        xd_result_2 = tf.squeeze(x_discriminator(x_fake))
        xd_fake_loss = tf.losses.log_loss(targets_fake, xd_result_2)
        
        _xd_train_loss = xd_real_loss + xd_fake_loss
    
    xd_grads = tape.gradient(_xd_train_loss, x_discriminator.trainable_variables)
    if int(global_step % LOG_FREQUENCY) == 0:
        summary_ops_v2.scalar(name='x_discriminator_real_loss', tensor=xd_real_loss,
                              step=global_step)
        summary_ops_v2.scalar(name='x_discriminator_fake_loss', tensor=xd_fake_loss,
                              step=global_step)
        summary_ops_v2.scalar(name='x_discriminator_loss', tensor=_xd_train_loss,
                              step=global_step)
        for grad, variable in zip(xd_grads, x_discriminator.trainable_variables):
            summary_ops_v2.histogram(name='gradients/' + variable.name, tensor=tf.math.l2_normalize(grad),
                                     step=global_step)
            summary_ops_v2.histogram(name='variables/' + variable.name, tensor=tf.math.l2_normalize(variable),
                                     step=global_step)
    optimizer.apply_gradients(zip(xd_grads, x_discriminator.trainable_variables),
                              global_step=global_step_xd)
    
    return _xd_train_loss


def _train_decoder_step(decoder: model.Decoder,
                        x_discriminator: model.XDiscriminator,
                        optimizer: tf.train.Optimizer,
                        targets: tf.Tensor,
                        global_step: tf.Variable,
                        global_step_decoder: tf.Variable,
                        z_generator: Callable[[], tf.Variable]) -> tf.Tensor:
    """
    Trains the decoder model for one step (one batch).
    
    :param decoder: instance of decoder model
    :param x_discriminator: instance of the x discriminator model
    :param optimizer: instance of chosen optimizer
    :param targets: target tensor for loss calculation
    :param global_step: the global step variable
    :param global_step_decoder: global step variable for decoder
    :param z_generator: callable function that returns a z variable
    :return: the calculated loss
    """
    with tf.GradientTape() as tape:
        z = z_generator()
        
        x_fake = decoder(z)
        xd_result = tf.squeeze(x_discriminator(x_fake))
        _decoder_train_loss = tf.losses.log_loss(targets, xd_result)
    
    grads = tape.gradient(_decoder_train_loss, decoder.trainable_variables)
    if int(global_step % LOG_FREQUENCY) == 0:
        summary_ops_v2.scalar(name='decoder_loss', tensor=_decoder_train_loss,
                              step=global_step)
        for grad, variable in zip(grads, decoder.trainable_variables):
            summary_ops_v2.histogram(name='gradients/' + variable.name, tensor=tf.math.l2_normalize(grad),
                                     step=global_step)
            summary_ops_v2.histogram(name='variables/' + variable.name, tensor=tf.math.l2_normalize(variable),
                                     step=global_step)
    optimizer.apply_gradients(zip(grads, decoder.trainable_variables),
                              global_step=global_step_decoder)
    
    return _decoder_train_loss


def _train_zdiscriminator_step(z_discriminator: model.ZDiscriminator,
                               encoder: model.Encoder,
                               optimizer: tf.train.Optimizer,
                               inputs: tf.Tensor,
                               targets_real: tf.Tensor,
                               targets_fake: tf.Tensor,
                               global_step: tf.Variable,
                               global_step_zd: tf.Variable,
                               z_generator: Callable[[], tf.Variable]) -> tf.Tensor:
    """
    Trains the z discriminator one step (one batch).
    
    :param z_discriminator: instance of z discriminator model
    :param encoder: instance of encoder model
    :param optimizer: instance of chosen optimizer
    :param inputs: inputs from dataset
    :param targets_real: target tensor for real loss calculation
    :param targets_fake: target tensor for fake loss calculation
    :param global_step: the global step variable
    :param global_step_zd: global step variable for zd
    :param z_generator: callable function that returns a z variable
    :return: the calculated loss
    """
    with tf.GradientTape() as tape:
        z = z_generator()
        
        zd_result = tf.squeeze(z_discriminator(z))
        zd_real_loss = tf.losses.log_loss(targets_real, zd_result)
        
        z = tf.squeeze(encoder(inputs))
        zd_result = tf.squeeze(z_discriminator(z))
        zd_fake_loss = tf.losses.log_loss(targets_fake, zd_result)
        
        _zd_train_loss = zd_real_loss + zd_fake_loss
    
    zd_grads = tape.gradient(_zd_train_loss, z_discriminator.trainable_variables)
    if int(global_step % LOG_FREQUENCY) == 0:
        summary_ops_v2.scalar(name='z_discriminator_real_loss', tensor=zd_real_loss,
                              step=global_step)
        summary_ops_v2.scalar(name='z_discriminator_fake_loss', tensor=zd_fake_loss,
                              step=global_step)
        summary_ops_v2.scalar(name='z_discriminator_loss', tensor=_zd_train_loss,
                              step=global_step)
        for grad, variable in zip(zd_grads, z_discriminator.trainable_variables):
            summary_ops_v2.histogram(name='gradients/' + variable.name, tensor=tf.math.l2_normalize(grad),
                                     step=global_step)
            summary_ops_v2.histogram(name='variables/' + variable.name, tensor=tf.math.l2_normalize(variable),
                                     step=global_step)
    optimizer.apply_gradients(zip(zd_grads, z_discriminator.trainable_variables),
                              global_step=global_step_zd)
    
    return _zd_train_loss


def _train_enc_dec_step(encoder: model.Encoder, decoder: model.Decoder,
                        z_discriminator: model.ZDiscriminator,
                        optimizer: tf.train.Optimizer,
                        inputs: tf.Tensor,
                        targets: tf.Tensor,
                        global_step: tf.Variable,
                        global_step_enc_dec: tf.Variable) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Trains the encoder and decoder jointly for one step (one batch).
    
    :param encoder: instance of encoder model
    :param decoder: instance of decoder model
    :param z_discriminator: instance of z discriminator model
    :param optimizer: instance of chosen optimizer
    :param inputs: inputs from dataset
    :param targets: target tensor for loss calculation
    :param global_step: the global step variable
    :param global_step_enc_dec: global step variable for enc_dec
    :return: tuple of encoder loss, reconstruction loss, reconstructed input
    """
    with tf.GradientTape() as tape:
        z = encoder(inputs)
        x_decoded = decoder(z)
        
        zd_result = tf.squeeze(z_discriminator(tf.squeeze(z)))
        encoder_loss = tf.losses.log_loss(targets, zd_result) * 2.0
        reconstruction_loss = tf.losses.log_loss(inputs, x_decoded)
        _enc_dec_train_loss = encoder_loss + reconstruction_loss
    
    enc_dec_grads = tape.gradient(_enc_dec_train_loss,
                                  encoder.trainable_variables + decoder.trainable_variables)
    if int(global_step % LOG_FREQUENCY) == 0:
        summary_ops_v2.scalar(name='encoder_loss', tensor=encoder_loss,
                              step=global_step)
        summary_ops_v2.scalar(name='reconstruction_loss', tensor=reconstruction_loss,
                              step=global_step)
        summary_ops_v2.scalar(name='encoder_decoder_loss', tensor=_enc_dec_train_loss,
                              step=global_step)
        for grad, variable in zip(enc_dec_grads, encoder.trainable_variables + decoder.trainable_variables):
            summary_ops_v2.histogram(name='gradients/' + variable.name, tensor=tf.math.l2_normalize(grad),
                                     step=global_step)
            summary_ops_v2.histogram(name='variables/' + variable.name, tensor=tf.math.l2_normalize(variable),
                                     step=global_step)
    optimizer.apply_gradients(zip(enc_dec_grads,
                                  encoder.trainable_variables + decoder.trainable_variables),
                              global_step=global_step_enc_dec)
    
    return encoder_loss, reconstruction_loss, x_decoded


def _get_z_variable(batch_size: int, zsize: int) -> tf.Variable:
    """
    Creates and returns a z variable taken from a normal distribution.
    
    :param batch_size: size of the batch
    :param zsize: size of the z latent space
    :return: created variable
    """
    z = K.reshape(K.random_normal((batch_size, zsize)), (-1, 1, 1, zsize))
    return K.variable(z)


def _normalize(feature: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Normalizes a tensor from a 0-255 range to a 0-1 range and adds one dimension.
    
    :param feature: tensor to be normalized
    :param label: label tensor
    :return: normalized tensor
    """
    return K.expand_dims(tf.divide(feature, 255.0)), label


if __name__ == "__main__":
    tf.enable_eager_execution()
    inlier_classes = [3]
    iteration = 1
    train_dataset, _ = prepare_training_data(test_fold_id=0, inlier_classes=inlier_classes,
                                             total_classes=10)
    train_summary_writer = summary_ops_v2.create_file_writer(
        './summaries/train/number-' + str(inlier_classes[0]) + '/' + str(iteration))
    with train_summary_writer.as_default():
        train(dataset=train_dataset, iteration=iteration,
              weights_prefix='weights/' + str(inlier_classes[0]) + '/')
