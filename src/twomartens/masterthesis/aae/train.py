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

"""aae.train.py: contains training functionality"""
import functools
import math
import os
import pickle
import time
from typing import Callable, Sequence, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import summary_ops_v2

from .model import Decoder, Encoder, XDiscriminator, ZDiscriminator
from .util import prepare_image

# shortcuts for tensorflow sub packages and classes
k = tf.keras.backend
AdamOptimizer = tf.train.AdamOptimizer
tfe = tf.contrib.eager
binary_crossentropy = tf.keras.losses.binary_crossentropy

GRACE = 10


def prepare_training_data(test_fold_id: int, inlier_classes: Sequence[int], total_classes: int,
                          fold_prefix: str = 'data/data_fold_',
                          batch_size: int = 128, folds: int = 5) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Prepares the MNIST data
    
    :param test_fold_id: id of test fold
    :param inlier_classes: list of class ids that are considered inliers
    :param total_classes: total number of classes
    :param fold_prefix: the prefix for the fold pickle files (default: 'data/data_fold_')
    :param batch_size: size of batch (default: 128)
    :param folds: number of folds (default: 5)
    :return: tuple(train dataset, valid dataset)
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
    
    def list_of_pairs_to_numpy(list_of_pairs: Sequence[Tuple[int, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts a list of pairs to a numpy array.

        :param list_of_pairs: list of pairs
        :return: numpy array
        """
        return np.asarray([x[1] for x in list_of_pairs], np.float32), np.asarray([x[0] for x in list_of_pairs], np.int)
    
    mnist_train_x, mnist_train_y = list_of_pairs_to_numpy(mnist_train)
    mnist_valid_x, mnist_valid_y = list_of_pairs_to_numpy(mnist_valid)
    
    # get dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((mnist_train_x, mnist_train_y))
    train_dataset = train_dataset.shuffle(mnist_train_x.shape[0]).batch(batch_size,
                                                                        drop_remainder=True).map(normalize)
    valid_dataset = tf.data.Dataset.from_tensor_slices((mnist_valid_x, mnist_valid_y))
    valid_dataset = valid_dataset.shuffle(mnist_valid_x.shape[0]).batch(batch_size,
                                                                        drop_remainder=True).map(normalize)
    
    return train_dataset, valid_dataset


def train(dataset: tf.data.Dataset, iteration: int, result_prefix: str,
          weights_prefix: str,
          channels: int = 1, zsize: int = 32, lr: float = 0.002,
          batch_size: int = 128, train_epoch: int = 80,
          verbose: bool = True) -> None:
    """
    Train AAE for given data set.
    
    :param dataset: train dataset
    :param iteration: identifier for the current training run
    :param result_prefix: prefix for result images
    :param weights_prefix: prefix for weights directory
    :param channels: number of channels in input image (default: 1)
    :param zsize: size of the intermediary z (default: 32)
    :param lr: initial learning rate (default: 0.002)
    :param batch_size: the size of each batch (default: 128)
    :param train_epoch: number of epochs to train (default: 80)
    :param verbose: if True prints train progress info to console (default: True)
    """
    
    # get models
    encoder = Encoder(zsize)
    decoder = Decoder(channels)
    z_discriminator = ZDiscriminator()
    x_discriminator = XDiscriminator()
    
    # define optimizers
    learning_rate_var = k.variable(lr)
    decoder_optimizer = AdamOptimizer(learning_rate=learning_rate_var, beta1=0.5, beta2=0.999)
    enc_dec_optimizer = AdamOptimizer(learning_rate=learning_rate_var, beta1=0.5, beta2=0.999)
    z_discriminator_optimizer = AdamOptimizer(learning_rate=learning_rate_var, beta1=0.5, beta2=0.999)
    x_discriminator_optimizer = AdamOptimizer(learning_rate=learning_rate_var, beta1=0.5, beta2=0.999)
    
    # train
    y_real = k.ones(batch_size)
    y_fake = k.zeros(batch_size)
    sample = k.expand_dims(k.expand_dims(k.random_normal((64, zsize)), axis=1), axis=1)

    z_generator = functools.partial(get_z_variable, batch_size=batch_size, zsize=zsize)

    global_step_decoder = k.variable(0, dtype=tf.int64)
    global_step_enc_dec = k.variable(0, dtype=tf.int64)
    global_step_xd = k.variable(0, dtype=tf.int64)
    global_step_zd = k.variable(0, dtype=tf.int64)
    
    encoder_lowest_loss = math.inf
    decoder_lowest_loss = math.inf
    enc_dec_lowest_loss = math.inf
    zd_lowest_loss = math.inf
    xd_lowest_loss = math.inf
    total_lowest_loss = math.inf
    grace_period = GRACE
    
    checkpoint_dir = os.path.join(weights_prefix, str(iteration) + '/')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    checkpoint = tf.train.Checkpoint(encoder=encoder,
                                     decoder=decoder,
                                     z_discriminator=z_discriminator,
                                     x_discriminator=x_discriminator,
                                     decoder_optimizer=decoder_optimizer,
                                     z_discriminator_optimizer=z_discriminator_optimizer,
                                     x_discriminator_optimizer=x_discriminator_optimizer,
                                     enc_dec_optimizer=enc_dec_optimizer,
                                     step_counter=global_step_decoder)
    if latest_checkpoint is not None:
        # if there is a checkpoint in the current training iteration, proceed from there
        checkpoint.restore(latest_checkpoint).assert_consumed()
    
    for epoch in range(train_epoch):
        # define loss variables
        encoder_loss_avg = tfe.metrics.Mean(name='encoder_loss', dtype=tf.float32)
        decoder_loss_avg = tfe.metrics.Mean(name='decoder_loss', dtype=tf.float32)
        enc_dec_loss_avg = tfe.metrics.Mean(name='encoder_decoder_loss', dtype=tf.float32)
        zd_loss_avg = tfe.metrics.Mean(name='z_discriminator_loss', dtype=tf.float32)
        xd_loss_avg = tfe.metrics.Mean(name='x_discriminator_loss', dtype=tf.float32)
        
        epoch_start_time = time.time()
        
        # update learning rate
        if (epoch + 1) % 30 == 0:
            learning_rate_var.assign(learning_rate_var.value() / 4)
            if verbose:
                print("learning rate change!")
        
        log_frequency = 10
        batch_iteration = k.variable(0, dtype=tf.int64)
        for x, _ in dataset:
            # x discriminator
            _xd_train_loss = train_xdiscriminator_step(x_discriminator=x_discriminator,
                                                       decoder=decoder,
                                                       optimizer=x_discriminator_optimizer,
                                                       inputs=x,
                                                       targets_real=y_real,
                                                       targets_fake=y_fake,
                                                       global_step=global_step_xd,
                                                       z_generator=z_generator)
            xd_loss_avg(_xd_train_loss)
            
            # --------
            # decoder
            _decoder_train_loss = train_decoder_step(decoder=decoder,
                                                     x_discriminator=x_discriminator,
                                                     optimizer=decoder_optimizer,
                                                     targets=y_real,
                                                     global_step=global_step_decoder,
                                                     z_generator=z_generator)
            decoder_loss_avg(_decoder_train_loss)
            
            # ---------
            # z discriminator
            _zd_train_loss = train_zdiscriminator_step(z_discriminator=z_discriminator,
                                                       encoder=encoder,
                                                       optimizer=z_discriminator_optimizer,
                                                       inputs=x,
                                                       targets_real=y_real,
                                                       targets_fake=y_fake,
                                                       global_step=global_step_zd,
                                                       z_generator=z_generator)
            zd_loss_avg(_zd_train_loss)
            
            # -----------
            # encoder + decoder
            encoder_loss, reconstruction_loss, x_decoded = train_enc_dec_step(encoder=encoder,
                                                                              decoder=decoder,
                                                                              z_discriminator=z_discriminator,
                                                                              optimizer=enc_dec_optimizer,
                                                                              inputs=x,
                                                                              targets=y_real,
                                                                              global_step=global_step_enc_dec)
            enc_dec_loss_avg(reconstruction_loss)
            encoder_loss_avg(encoder_loss)

            if int(global_step_decoder % log_frequency) == 0:
                # log the losses every log frequency batches
                summary_ops_v2.scalar('encoder_loss', encoder_loss_avg.result(False), step=global_step_enc_dec)
                summary_ops_v2.scalar('decoder_loss', decoder_loss_avg.result(False), step=global_step_decoder)
                summary_ops_v2.scalar('encoder_decoder_loss', enc_dec_loss_avg.result(False), step=global_step_enc_dec)
                summary_ops_v2.scalar('z_discriminator_loss', zd_loss_avg.result(False), step=global_step_zd)
                summary_ops_v2.scalar('x_discriminator_loss', xd_loss_avg.result(False), step=global_step_xd)

            if int(batch_iteration) == 0:
                directory = 'results' + str(inlier_classes[0])
                if not os.path.exists(directory):
                    os.makedirs(directory)
                comparison = k.concatenate([x[:64], x_decoded[:64]], axis=0)
                grid = prepare_image(comparison.cpu(), nrow=64)
                summary_ops_v2.image(name='reconstruction_' + str(epoch),
                                     tensor=k.expand_dims(grid, axis=0), max_images=1,
                                     step=global_step_decoder)
                from PIL import Image
                filename = os.path.join(result_prefix, 'reconstruction_' + str(epoch) + '.png')
                ndarr = grid.cpu().numpy()
                im = Image.fromarray(ndarr)
                im.save(filename)

            batch_iteration.assign_add(1)
        
        epoch_end_time = time.time()
        per_epoch_time = epoch_end_time - epoch_start_time
        
        # final losses of epoch
        decoder_loss = decoder_loss_avg.result(False)
        encoder_loss = encoder_loss_avg.result(False)
        enc_dec_loss = enc_dec_loss_avg.result(False)
        xd_loss = xd_loss_avg.result(False)
        zd_loss = zd_loss_avg.result(False)
        if verbose:
            print((
                f"[{epoch + 1:d}/{train_epoch:d}] - "
                f"train time: {per_epoch_time:.2f}, "
                f"Decoder loss: {decoder_loss:.3f}, "
                f"X Discriminator loss: {xd_loss:.3f}, "
                f"Z Discriminator loss: {zd_loss:.3f}, "
                f"Encoder + Decoder loss: {enc_dec_loss:.3f}, "
                f"Encoder loss: {encoder_loss:.3f}"
            ))
        
        # save sample image
        resultsample = decoder(sample).cpu()
        directory = 'results' + str(inlier_classes[0])
        os.makedirs(directory, exist_ok=True)
        grid = prepare_image(resultsample)
        summary_ops_v2.image(name='sample_' + str(epoch), tensor=k.expand_dims(grid, axis=0),
                             max_images=1, step=global_step_decoder)
        from PIL import Image
        filename = os.path.join(result_prefix, 'sample_' + str(epoch) + '.png')
        ndarr = grid.cpu().numpy()
        im = Image.fromarray(ndarr)
        im.save(filename)
        
        # save weights at end of epoch
        checkpoint.save(checkpoint_prefix)

        # check for improvements in error reduction - otherwise early stopping
        strike = False
        total_strike = False
        total_loss = encoder_loss + decoder_loss + enc_dec_loss + xd_loss + zd_loss
        if total_loss < total_lowest_loss:
            total_lowest_loss = total_loss
        elif total_loss > 6:
            total_strike = True
        if encoder_loss < encoder_lowest_loss:
            encoder_lowest_loss = encoder_loss
        else:
            strike = True
        if decoder_loss < decoder_lowest_loss:
            decoder_lowest_loss = decoder_loss
        else:
            strike = True
        if enc_dec_loss < enc_dec_lowest_loss:
            enc_dec_lowest_loss = enc_dec_loss
        else:
            strike = True
        if xd_loss < xd_lowest_loss:
            xd_lowest_loss = xd_loss
        else:
            strike = True
        if zd_loss < zd_lowest_loss:
            zd_lowest_loss = zd_loss
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


def train_xdiscriminator_step(x_discriminator: XDiscriminator, decoder: Decoder,
                              optimizer: tf.train.Optimizer,
                              inputs: tf.Tensor, targets_real: tf.Tensor,
                              targets_fake: tf.Tensor, global_step: tf.Variable,
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
    :param z_generator: callable function that returns a z variable
    :return: the calculated loss
    """
    with tf.GradientTape() as tape:
        xd_result_1 = tf.squeeze(x_discriminator(inputs))
        xd_real_loss = binary_crossentropy(targets_real, xd_result_1)
        
        z = z_generator()
        x_fake = decoder(z)
        xd_result_2 = tf.squeeze(x_discriminator(x_fake))
        xd_fake_loss = binary_crossentropy(targets_fake, xd_result_2)
        
        _xd_train_loss = xd_real_loss + xd_fake_loss
    
    xd_grads = tape.gradient(_xd_train_loss, x_discriminator.trainable_variables)
    optimizer.apply_gradients(zip(xd_grads, x_discriminator.trainable_variables),
                              global_step=global_step)
    
    return _xd_train_loss


def train_decoder_step(decoder: Decoder, x_discriminator: XDiscriminator,
                       optimizer: tf.train.Optimizer,
                       targets: tf.Tensor, global_step: tf.Variable,
                       z_generator: Callable[[], tf.Variable]) -> tf.Tensor:
    """
    Trains the decoder model for one step (one batch).
    
    :param decoder: instance of decoder model
    :param x_discriminator: instance of the x discriminator model
    :param optimizer: instance of chosen optimizer
    :param targets: target tensor for loss calculation
    :param global_step: the global step variable
    :param z_generator: callable function that returns a z variable
    :return: the calculated loss
    """
    with tf.GradientTape() as tape:
        z = z_generator()
        
        x_fake = decoder(z)
        xd_result = tf.squeeze(x_discriminator(x_fake))
        _decoder_train_loss = binary_crossentropy(targets, xd_result)
    
    grads = tape.gradient(_decoder_train_loss, decoder.trainable_variables)
    optimizer.apply_gradients(zip(grads, decoder.trainable_variables),
                              global_step=global_step)
    
    return _decoder_train_loss


def train_zdiscriminator_step(z_discriminator: ZDiscriminator, encoder: Encoder,
                              optimizer: tf.train.Optimizer,
                              inputs: tf.Tensor, targets_real: tf.Tensor,
                              targets_fake: tf.Tensor, global_step: tf.Variable,
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
    :param z_generator: callable function that returns a z variable
    :return: the calculated loss
    """
    with tf.GradientTape() as tape:
        z = z_generator()
        
        zd_result = tf.squeeze(z_discriminator(z))
        zd_real_loss = binary_crossentropy(targets_real, zd_result)
        
        z = tf.squeeze(encoder(inputs))
        zd_result = tf.squeeze(z_discriminator(z))
        zd_fake_loss = binary_crossentropy(targets_fake, zd_result)
        
        _zd_train_loss = zd_real_loss + zd_fake_loss
    
    zd_grads = tape.gradient(_zd_train_loss, z_discriminator.trainable_variables)
    optimizer.apply_gradients(zip(zd_grads, z_discriminator.trainable_variables),
                              global_step=global_step)
    
    return _zd_train_loss


def train_enc_dec_step(encoder: Encoder, decoder: Decoder, z_discriminator: ZDiscriminator,
                       optimizer: tf.train.Optimizer, inputs: tf.Tensor,
                       targets: tf.Tensor, global_step: tf.Variable) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Trains the encoder and decoder jointly for one step (one batch).
    
    :param encoder: instance of encoder model
    :param decoder: instance of decoder model
    :param z_discriminator: instance of z discriminator model
    :param optimizer: instance of chosen optimizer
    :param inputs: inputs from dataset
    :param targets: target tensor for loss calculation
    :param global_step: the global step variable
    :return: tuple of encoder loss, reconstruction loss, reconstructed input
    """
    with tf.GradientTape() as tape:
        z = encoder(inputs)
        x_decoded = decoder(z)
        
        zd_result = tf.squeeze(z_discriminator(tf.squeeze(z)))
        encoder_loss = binary_crossentropy(targets, zd_result) * 2.0
        reconstruction_loss = binary_crossentropy(inputs, x_decoded)
        _enc_dec_train_loss = encoder_loss + reconstruction_loss
    
    enc_dec_grads = tape.gradient(_enc_dec_train_loss,
                                  encoder.trainable_variables + decoder.trainable_variables)
    optimizer.apply_gradients(zip(enc_dec_grads,
                                  encoder.trainable_variables + decoder.trainable_variables),
                              global_step=global_step)
    
    return encoder_loss, reconstruction_loss, x_decoded


def get_z_variable(batch_size: int, zsize: int) -> tf.Variable:
    """
    Creates and returns a z variable taken from a normal distribution.
    
    :param batch_size: size of the batch
    :param zsize: size of the z latent space
    :return: created variable
    """
    z = k.reshape(k.random_normal((batch_size, zsize)), (-1, 1, 1, zsize))
    return k.variable(z)


def normalize(feature: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Normalizes a tensor from a 0-255 range to a 0-1 range and adds one dimension.
    
    :param feature: tensor to be normalized
    :param label: label tensor
    :return: normalized tensor
    """
    return k.expand_dims(tf.divide(feature, 255.0)), label


def extract_batch(data: np.ndarray, it: int, batch_size: int) -> tfe.Variable:
    """
    Extracts a batch from data.
    
    :param data: numpy array of data
    :param it: current iteration in epoch (or batch number)
    :param batch_size: size of batch
    :return: tensor
    """
    x = data[it * batch_size:(it + 1) * batch_size, :, :] / 255.0
    return k.variable(x)


if __name__ == "__main__":
    tf.enable_eager_execution()
    inlier_classes = [0]
    iteration = 1
    train_dataset, _ = prepare_training_data(test_fold_id=0, inlier_classes=[0],
                                             total_classes=10)
    train_summary_writer = summary_ops_v2.create_file_writer(
        './summaries/train/number-' + str(inlier_classes[0]) + '/' + str(iteration))
    with train_summary_writer.as_default(), summary_ops_v2.always_record_summaries():
        train(dataset=train_dataset, iteration=iteration,
              result_prefix='results' + str(inlier_classes[0]) + '/',
              weights_prefix='weights/' + str(inlier_classes[0]) + '/')
