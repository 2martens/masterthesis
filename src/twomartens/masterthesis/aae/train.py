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
import os
import pickle
import random
import time
from typing import Sequence, Tuple

import numpy as np
import tensorflow as tf

from .model import Decoder, Encoder, XDiscriminator, ZDiscriminator
from .util import save_image

# shortcuts for tensorflow sub packages and classes
k = tf.keras.backend
AdamOptimizer = tf.train.AdamOptimizer
tfe = tf.contrib.eager
binary_crossentropy = tf.keras.losses.binary_crossentropy


def train_mnist(folding_id: int, inlier_classes: Sequence[int], total_classes: int,
                channels: int = 1, zsize: int = 32, lr: float = 0.002,
                batch_size: int = 128, train_epoch: int = 80,
                folds: int = 5):
    """
    Train AAE for mnist data set.
    :param folding_id: id of fold used for test data
    :param inlier_classes: list of class ids that are considered inliers
    :param total_classes: total number of classes
    :param channels: number of channels in input image
    :param zsize: size of the intermediary z
    :param lr: learning rate
    :param batch_size: size of each batch
    :param train_epoch: number of epochs to train
    :param folds: number of folds available
    """
    # prepare data
    mnist_train = []
    mnist_valid = []
    
    for i in range(folds):
        if i != folding_id:
            with open('data/data_fold_%d.pkl' % i, 'rb') as pkl:
                fold = pickle.load(pkl)
            if len(mnist_valid) == 0:
                mnist_valid = fold
            else:
                mnist_train += fold
    
    outlier_classes = []
    for i in range(total_classes):
        if i not in inlier_classes:
            outlier_classes.append(i)
    
    # keep only train classes
    mnist_train = [x for x in mnist_train if x[0] in inlier_classes]
    random.shuffle(mnist_train)
    
    def list_of_pairs_to_numpy(list_of_pairs: Sequence[Tuple[int, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts a list of pairs to a numpy array.
        :param list_of_pairs: list of pairs
        :return: numpy array
        """
        return np.asarray([x[1] for x in list_of_pairs], np.float32), np.asarray([x[0] for x in list_of_pairs], np.int)
    
    mnist_train_x, mnist_train_y = list_of_pairs_to_numpy(mnist_train)
    
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
    y_real_z = k.ones(batch_size)
    y_fake_z = k.zeros(batch_size)
    sample = k.expand_dims(k.expand_dims(k.random_normal((64, zsize)), axis=1), axis=1)
    global_step = tf.train.get_or_create_global_step()
    
    encoder_loss_history = []
    decoder_loss_history = []
    enc_dec_loss_history = []
    zd_loss_history = []
    xd_loss_history = []
    
    for epoch in range(train_epoch):
        # define loss variables
        encoder_loss_avg = tfe.metrics.Mean(name='encoder_loss', dtype=tf.float32)
        decoder_loss_avg = tfe.metrics.Mean(name='decoder_loss', dtype=tf.float32)
        enc_dec_loss_avg = tfe.metrics.Mean(name='enc_dec_loss', dtype=tf.float32)
        zd_loss_avg = tfe.metrics.Mean(name='zd_loss', dtype=tf.float32)
        xd_loss_avg = tfe.metrics.Mean(name='xd_loss', dtype=tf.float32)
        
        epoch_start_time = time.time()
        
        def shuffle(train_data: np.ndarray) -> None:
            """
            Shuffles the given training data inplace.
            :param train_data: numpy array of training data
            """
            np.take(train_data, np.random.permutation(train_data.shape[0]), axis=0, out=train_data)
        
        shuffle(mnist_train_x)
        
        # update learning rate
        if (epoch + 1) % 30 == 0:
            learning_rate_var.assign(learning_rate_var.value() / 4)
            print("learning rate change!")
        
        nr_batches = len(mnist_train_x) // batch_size
        for it in range(nr_batches):
            x = k.expand_dims(extract_batch(mnist_train_x, it, batch_size))
            # x discriminator
            with tf.GradientTape() as tape:
                xd_result = tf.squeeze(x_discriminator(x))
                xd_real_loss = binary_crossentropy(y_real, xd_result)
                z = k.reshape(k.random_normal((batch_size, zsize)), (-1, 1, 1, zsize))
                z = k.variable(z)
                
                x_fake = decoder(z)
                xd_result = tf.squeeze(x_discriminator(x_fake))
                xd_fake_loss = binary_crossentropy(y_fake, xd_result)
                
                _xd_train_loss = xd_real_loss + xd_fake_loss
            
            xd_grads = tape.gradient(_xd_train_loss, x_discriminator.trainable_variables)
            x_discriminator_optimizer.apply_gradients(zip(xd_grads, x_discriminator.trainable_variables),
                                                      global_step=global_step)
            xd_loss_avg(_xd_train_loss)
            
            # --------
            # decoder
            with tf.GradientTape() as tape:
                z = k.reshape(k.random_normal((batch_size, zsize)), (-1, 1, 1, zsize))
                z = k.variable(z)
                
                x_fake = decoder(z)
                xd_result = tf.squeeze(x_discriminator(x_fake))
                _decoder_train_loss = binary_crossentropy(y_real, xd_result)
            
            decoder_grads = tape.gradient(_decoder_train_loss, decoder.trainable_variables)
            decoder_optimizer.apply_gradients(zip(decoder_grads, decoder.trainable_variables),
                                              global_step=global_step)
            decoder_loss_avg(_decoder_train_loss)
            
            # ---------
            # z discriminator
            with tf.GradientTape() as tape:
                z = k.reshape(k.random_normal((batch_size, zsize)), (-1, zsize))
                z = k.variable(z)
                
                zd_result = tf.squeeze(z_discriminator(z))
                zd_real_loss = binary_crossentropy(y_real_z, zd_result)
                
                z = tf.squeeze(encoder(x))
                zd_result = tf.squeeze(z_discriminator(z))
                zd_fake_loss = binary_crossentropy(y_fake_z, zd_result)
                
                _zd_train_loss = zd_real_loss + zd_fake_loss
            
            zd_grads = tape.gradient(_zd_train_loss, z_discriminator.trainable_variables)
            z_discriminator_optimizer.apply_gradients(zip(zd_grads, z_discriminator.trainable_variables),
                                                      global_step=global_step)
            zd_loss_avg(_zd_train_loss)
            
            # -----------
            # encoder + decoder
            with tf.GradientTape() as tape:
                z = encoder(x)
                x_decoded = decoder(z)
                
                zd_result = tf.squeeze(z_discriminator(tf.squeeze(z)))
                encoder_loss = binary_crossentropy(y_real_z, zd_result) * 2.0
                recovery_loss = binary_crossentropy(x, x_decoded)
                _enc_dec_train_loss = encoder_loss + recovery_loss
            
            enc_dec_grads = tape.gradient(_enc_dec_train_loss,
                                          encoder.trainable_variables + decoder.trainable_variables)
            enc_dec_optimizer.apply_gradients(zip(enc_dec_grads,
                                                  encoder.trainable_variables + decoder.trainable_variables))
            enc_dec_loss_avg(recovery_loss)
            encoder_loss_avg(encoder_loss)
            
            if it == 0:
                directory = 'results' + str(inlier_classes[0])
                if not os.path.exists(directory):
                    os.makedirs(directory)
                comparison = k.concatenate([x[:64], x_decoded[:64]], axis=0)
                save_image(comparison.cpu(),
                           'results' + str(inlier_classes[0]) + '/reconstruction_' + str(epoch) + '.png', nrow=64)
        
        encoder_loss_history.append(encoder_loss_avg.result())
        decoder_loss_history.append(decoder_loss_avg.result())
        enc_dec_loss_history.append(enc_dec_loss_avg.result())
        xd_loss_history.append(xd_loss_avg.result())
        zd_loss_history.append(zd_loss_avg.result())
        
        epoch_end_time = time.time()
        per_epoch_time = epoch_end_time - epoch_start_time
        
        print((
            f"[{epoch + 1:d}/{train_epoch:d}] - "
            f"train time: {per_epoch_time:.2f}, "
            f"Decoder loss: {decoder_loss_avg.result()}, X Discriminator loss: {xd_loss_avg.result():.3f}, "
            f"Z Discriminator loss: {zd_loss_avg.result():.3f}, "
            f"Encoder + Decoder loss: {enc_dec_loss_avg.result():.3f}, "
            f"Encoder loss: {encoder_loss_avg.result():.3f}"
        ))
        
        # save sample image
        resultsample = decoder(sample).cpu()
        directory = 'results' + str(inlier_classes[0])
        os.makedirs(directory, exist_ok=True)
        save_image(resultsample,
                   'results' + str(inlier_classes[0]) + '/sample_' + str(epoch) + '.png')
    
    print("Training finish!... save training results")
    encoder.save_weights("./weights/encoder")
    decoder.save_weights("./weights/decoder")
    z_discriminator.save_weights("./weights/z_discriminator")
    x_discriminator.save_weights("./weights/x_discriminator")
    with open("./results0/losses/encoder_loss.txt", "wb") as file:
        pickle.dump(encoder_loss_history, file)
    with open("./results0/losses/decoder_loss.txt", "wb") as file:
        pickle.dump(decoder_loss_history, file)
    with open("./results0/losses/enc_dec_loss.txt", "wb") as file:
        pickle.dump(enc_dec_loss_history, file)
    with open("./results0/losses/xd_loss.txt", "wb") as file:
        pickle.dump(xd_loss_history, file)
    with open("./results0/losses/zd_loss.txt", "wb") as file:
        pickle.dump(zd_loss_history, file)


def extract_batch(data: np.ndarray, it: int, batch_size: int) -> tf.Variable:
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
    train_mnist(folding_id=0, inlier_classes=[0], total_classes=10)
