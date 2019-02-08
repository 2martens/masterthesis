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
Provides the models of my AAE implementation.

Classes:
    ``Encoder``: encodes an image input to a latent space
    
    ``Decoder``: decodes data from a latent space to resemble input data
    
    ``XDiscriminator``: differentiates between real input data and decoded input data
    
    ``ZDiscriminator``: differentiates between z values drawn from a normal distribution (real) and the encoded input
    (fake)

"""
import tensorflow as tf

# shortcuts for tensorflow - quasi imports
keras = tf.keras
k = tf.keras.backend


class Encoder(keras.Model):
    """
    Encodes input to a latent space.
    
    Args:
        zsize: size of the latent space
    """
    
    def __init__(self, zsize: int) -> None:
        super().__init__(name='encoder')
        weight_init = keras.initializers.RandomNormal(mean=0, stddev=0.02)
        self.x_padded = keras.layers.ZeroPadding2D(padding=1)
        self.conv1 = keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, name='conv1',
                                         padding='valid', kernel_initializer=weight_init)
        self.conv1_a = keras.layers.LeakyReLU(alpha=0.2)
        self.conv1_a_padded = keras.layers.ZeroPadding2D(padding=1)
        self.conv2 = keras.layers.Conv2D(filters=256, kernel_size=4, strides=2, name='conv2',
                                         padding='valid', kernel_initializer=weight_init)
        self.conv2_bn = keras.layers.BatchNormalization()
        self.conv2_a = keras.layers.LeakyReLU(alpha=0.2)
        self.conv2_a_padded = keras.layers.ZeroPadding2D(padding=1)
        self.conv3 = keras.layers.Conv2D(filters=512, kernel_size=4, strides=2, name='conv3',
                                         padding='valid', kernel_initializer=weight_init)
        self.conv3_bn = keras.layers.BatchNormalization()
        self.conv3_a = keras.layers.LeakyReLU(alpha=0.2)
        self.conv4 = keras.layers.Conv2D(filters=zsize, kernel_size=4, strides=1, name='conv4',
                                         padding='valid', kernel_initializer=weight_init)
    
    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """See base class."""
        result = self.x_padded(inputs)
        result = self.conv1(result)
        result = self.conv1_a(result)
        result = self.conv1_a_padded(result)
        result = self.conv2(result)
        result = self.conv2_bn(result)
        result = self.conv2_a(result)
        result = self.conv2_a_padded(result)
        result = self.conv3(result)
        result = self.conv3_bn(result)
        result = self.conv3_a(result)
        result = self.conv4(result)
        
        return result


class Decoder(keras.Model):
    """
    Generates input data from latent space values.
    
    Args:
        channels: number of channels in the input image
    """
    
    def __init__(self, channels: int) -> None:
        super().__init__(name='decoder')
        weight_init = keras.initializers.RandomNormal(mean=0, stddev=0.02)
        self.deconv1 = keras.layers.Conv2DTranspose(filters=256, kernel_size=4, strides=1, name='deconv1',
                                                    padding='valid', kernel_initializer=weight_init)
        self.deconv1_bn = keras.layers.BatchNormalization()
        self.deconv1_a = keras.layers.ReLU()
        self.deconv2 = keras.layers.Conv2DTranspose(filters=256, kernel_size=4, strides=2, name='deconv2',
                                                    padding='valid', kernel_initializer=weight_init)
        self.deconv2_cropped = keras.layers.Cropping2D(cropping=1)
        self.deconv2_bn = keras.layers.BatchNormalization()
        self.deconv2_a = keras.layers.ReLU()
        self.deconv3 = keras.layers.Conv2DTranspose(filters=128, kernel_size=4, strides=2, name='deconv3',
                                                    padding='valid', kernel_initializer=weight_init)
        self.deconv3_cropped = keras.layers.Cropping2D(cropping=1)
        self.deconv3_bn = keras.layers.BatchNormalization()
        self.deconv3_a = keras.layers.ReLU()
        self.deconv4 = keras.layers.Conv2DTranspose(filters=channels, kernel_size=4, strides=2, name='deconv4',
                                                    padding='valid', kernel_initializer=weight_init)
        self.deconv4_cropped = keras.layers.Cropping2D(cropping=1)
    
    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """See base class."""
        result = self.deconv1(inputs)
        result = self.deconv1_bn(result)
        result = self.deconv1_a(result)
        result = self.deconv2(result)
        result = self.deconv2_cropped(result)
        result = self.deconv2_bn(result)
        result = self.deconv2_a(result)
        result = self.deconv3(result)
        result = self.deconv3_cropped(result)
        result = self.deconv3_bn(result)
        result = self.deconv3_a(result)
        result = self.deconv4(result)
        result = self.deconv4_cropped(result)
        result = k.tanh(result) * 0.5 + 0.5
        
        return result


class ZDiscriminator(keras.Model):
    """
    Discriminates between encoded inputs and latent space distribution.
    
    The latent space value is drawn from a normal distribution with ``0`` mean
    and a variance of ``1``.
    """
    
    def __init__(self) -> None:
        super().__init__(name='zdiscriminator')
        weight_init = keras.initializers.RandomNormal(mean=0, stddev=0.02)
        self.zd1 = keras.layers.Dense(units=128, name='zd1', kernel_initializer=weight_init)
        self.zd1_a = keras.layers.LeakyReLU(alpha=0.2)
        self.zd2 = keras.layers.Dense(units=128, name='zd2', kernel_initializer=weight_init)
        self.zd2_a = keras.layers.LeakyReLU(alpha=0.2)
        self.zd3 = keras.layers.Dense(units=1, name='zd3', activation='sigmoid',
                                      kernel_initializer=weight_init)
    
    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """See base class."""
        result = self.zd1(inputs)
        result = self.zd1_a(result)
        result = self.zd2(result)
        result = self.zd2_a(result)
        result = self.zd3(result)
        
        return result


class XDiscriminator(keras.Model):
    """
    Discriminates between generated inputs and the actual inputs.
    """
    
    def __init__(self) -> None:
        super().__init__(name='xdiscriminator')
        weight_init = keras.initializers.RandomNormal(mean=0, stddev=0.02)
        self.x_padded = keras.layers.ZeroPadding2D(padding=1)
        self.xd1 = keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, name='xd1',
                                       padding='valid', kernel_initializer=weight_init)
        self.xd1_a = keras.layers.LeakyReLU(alpha=0.2)
        self.xd1_a_padded = keras.layers.ZeroPadding2D(padding=1)
        self.xd2 = keras.layers.Conv2D(filters=256, kernel_size=4, strides=2, name='xd2',
                                       padding='valid', kernel_initializer=weight_init)
        self.xd2_bn = keras.layers.BatchNormalization()
        self.xd2_a = keras.layers.LeakyReLU(alpha=0.2)
        self.xd2_a_padded = keras.layers.ZeroPadding2D(padding=1)
        self.xd3 = keras.layers.Conv2D(filters=512, kernel_size=4, strides=2, name='xd3',
                                       padding='valid', kernel_initializer=weight_init)
        self.xd3_bn = keras.layers.BatchNormalization()
        self.xd3_a = keras.layers.LeakyReLU(alpha=0.2)
        self.xd4 = keras.layers.Conv2D(filters=1, kernel_size=4, strides=1, name='xd4',
                                       padding='valid', kernel_initializer=weight_init,
                                       activation='sigmoid')
    
    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """See base class."""
        result = self.x_padded(inputs)
        result = self.xd1(result)
        result = self.xd1_a(result)
        result = self.xd1_a_padded(result)
        result = self.xd2(result)
        result = self.xd2_bn(result)
        result = self.xd2_a(result)
        result = self.xd2_a_padded(result)
        result = self.xd3(result)
        result = self.xd3_bn(result)
        result = self.xd3_a(result)
        result = self.xd4(result)
        
        return result
