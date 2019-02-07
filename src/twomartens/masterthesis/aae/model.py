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

"""aae.model.py: contains model definitions"""
import tensorflow as tf

# shortcuts for tensorflow - quasi imports
keras = tf.keras
k = tf.keras.backend
Model = keras.Model
sigmoid = keras.activations.sigmoid
RandomNormal = keras.initializers.RandomNormal
BatchNormalization = keras.layers.BatchNormalization
Conv2D = keras.layers.Conv2D
Conv2DTranspose = keras.layers.Conv2DTranspose
Dense = keras.layers.Dense
Cropping2D = keras.layers.Cropping2D
ZeroPadding2D = keras.layers.ZeroPadding2D
ReLU = keras.layers.ReLU
LeakyReLU = keras.layers.LeakyReLU


class Encoder(Model):
    """
    Encoder model.
    """
    
    def __init__(self, zsize: int) -> None:
        super().__init__(name='encoder')
        weight_init = RandomNormal(mean=0, stddev=0.02)
        self.x_padded = ZeroPadding2D(padding=1)
        self.conv1 = Conv2D(filters=64, kernel_size=4, strides=2, name='conv1',
                            padding='valid', kernel_initializer=weight_init)
        self.conv1_a = LeakyReLU(alpha=0.2)
        self.conv1_a_padded = ZeroPadding2D(padding=1)
        self.conv2 = Conv2D(filters=256, kernel_size=4, strides=2, name='conv2',
                            padding='valid', kernel_initializer=weight_init)
        self.conv2_bn = BatchNormalization()
        self.conv2_a = LeakyReLU(alpha=0.2)
        self.conv2_a_padded = ZeroPadding2D(padding=1)
        self.conv3 = Conv2D(filters=512, kernel_size=4, strides=2, name='conv3',
                            padding='valid', kernel_initializer=weight_init)
        self.conv3_bn = BatchNormalization()
        self.conv3_a = LeakyReLU(alpha=0.2)
        self.conv4 = Conv2D(filters=zsize, kernel_size=4, strides=1, name='conv4',
                            padding='valid', kernel_initializer=weight_init)
    
    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Performs the forward pass.
        :param inputs: input values
        :param kwargs: additional keyword arguments - none are used
        :return: result values
        """
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


class Decoder(Model):
    """
    Decoder model.
    """
    
    def __init__(self, channels: int) -> None:
        super().__init__(name='decoder')
        weight_init = RandomNormal(mean=0, stddev=0.02)
        self.deconv1 = Conv2DTranspose(filters=256, kernel_size=4, strides=1, name='deconv1',
                                       padding='valid', kernel_initializer=weight_init)
        self.deconv1_bn = BatchNormalization()
        self.deconv1_a = ReLU()
        self.deconv2 = Conv2DTranspose(filters=256, kernel_size=4, strides=2, name='deconv2',
                                       padding='valid', kernel_initializer=weight_init)
        self.deconv2_cropped = Cropping2D(cropping=1)
        self.deconv2_bn = BatchNormalization()
        self.deconv2_a = ReLU()
        self.deconv3 = Conv2DTranspose(filters=128, kernel_size=4, strides=2, name='deconv3',
                                       padding='valid', kernel_initializer=weight_init)
        self.deconv3_cropped = Cropping2D(cropping=1)
        self.deconv3_bn = BatchNormalization()
        self.deconv3_a = ReLU()
        self.deconv4 = Conv2DTranspose(filters=channels, kernel_size=4, strides=2, name='deconv4',
                                       padding='valid', kernel_initializer=weight_init)
        self.deconv4_cropped = Cropping2D(cropping=1)
    
    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Performs the forward pass.
        :param inputs: input values
        :param kwargs: additional keyword arguments - none are used
        :return: result values
        """
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


class ZDiscriminator(Model):
    """
    ZDiscriminator model
    """
    
    def __init__(self) -> None:
        super().__init__(name='zdiscriminator')
        weight_init = RandomNormal(mean=0, stddev=0.02)
        self.zd1 = Dense(units=128, name='zd1', kernel_initializer=weight_init)
        self.zd1_a = LeakyReLU(alpha=0.2)
        self.zd2 = Dense(units=128, name='zd2', kernel_initializer=weight_init)
        self.zd2_a = LeakyReLU(alpha=0.2)
        self.zd3 = Dense(units=1, name='zd3', activation='sigmoid',
                         kernel_initializer=weight_init)
    
    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Performs the forward pass.
        :param inputs: input values
        :param kwargs: additional keyword arguments - none are used
        :return: result values
        """
        result = self.zd1(inputs)
        result = self.zd1_a(result)
        result = self.zd2(result)
        result = self.zd2_a(result)
        result = self.zd3(result)
        
        return result


class XDiscriminator(Model):
    """
    XDiscriminator model
    """
    
    def __init__(self) -> None:
        super().__init__(name='xdiscriminator')
        weight_init = RandomNormal(mean=0, stddev=0.02)
        self.x_padded = ZeroPadding2D(padding=1)
        self.xd1 = Conv2D(filters=64, kernel_size=4, strides=2, name='xd1',
                          padding='valid', kernel_initializer=weight_init)
        self.xd1_a = LeakyReLU(alpha=0.2)
        self.xd1_a_padded = ZeroPadding2D(padding=1)
        self.xd2 = Conv2D(filters=256, kernel_size=4, strides=2, name='xd2',
                          padding='valid', kernel_initializer=weight_init)
        self.xd2_bn = BatchNormalization()
        self.xd2_a = LeakyReLU(alpha=0.2)
        self.xd2_a_padded = ZeroPadding2D(padding=1)
        self.xd3 = Conv2D(filters=512, kernel_size=4, strides=2, name='xd3',
                          padding='valid', kernel_initializer=weight_init)
        self.xd3_bn = BatchNormalization()
        self.xd3_a = LeakyReLU(alpha=0.2)
        self.xd4 = Conv2D(filters=1, kernel_size=4, strides=1, name='xd4',
                          padding='valid', kernel_initializer=weight_init,
                          activation='sigmoid')
    
    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Performs the forward pass.
        :param inputs: input values
        :param kwargs: additional keyword arguments - none are used
        :return: result values
        """
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
