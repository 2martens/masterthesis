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
Provides functionality to use the SSD Keras implementation.

Attributes:
    IMAGE_SIZE: tuple of (height, width, channels)
    N_CLASSES: number of known classes (without background)
    DROPOUT_RATE: rate for dropping weights
    IOU_THRESHOLD: threshold for required overlap with ground truth bounding box
    TOP_K: maximum number of predictions kept for each batch item after non-maximum suppression
    
Classes:
    ``DropoutSSD``: wraps Dropout SSD 300 model
    
    ``SSD``: wraps vanilla SSD 300 model
"""
from typing import Optional

import tensorflow as tf

from twomartens.masterthesis.ssd_keras.models import keras_ssd300
from twomartens.masterthesis.ssd_keras.models import keras_ssd300_dropout

IMAGE_SIZE = (240, 320, 3)  # TODO check with SceneNet RGB-D
N_CLASSES = 80
DROPOUT_RATE = 0.5
IOU_THRESHOLD = 0.45
TOP_K = 200


class SSD:
    """
    Wraps vanilla SSD 300 model.
    
    Args:
        mode: one of training, inference, and inference_fast
        weights_path: path to trained weights
    """
    
    def __init__(self, mode: str, weights_path: Optional[str] = None) -> None:
        self._model = keras_ssd300.ssd_300(image_size=IMAGE_SIZE, n_classes=N_CLASSES,
                                           mode=mode, iou_threshold=IOU_THRESHOLD, top_k=TOP_K)
        self.mode = mode
        
        # load existing weights
        if weights_path is not None:
            self._model.load_weights(weights_path, by_name=True)
    
    def __call__(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        return self._model(inputs)


class DropoutSSD:
    """
    Wraps Dropout SSD 300 model.
    
    Args:
        mode: one of training, inference, and inference_fast
        weights_path: path to trained weights
    """
    
    def __init__(self, mode: str, weights_path: Optional[str] = None) -> None:
        self._model = keras_ssd300_dropout.ssd_300_dropout(image_size=IMAGE_SIZE, n_classes=N_CLASSES,
                                                           dropout_rate=DROPOUT_RATE, mode=mode,
                                                           iou_threshold=IOU_THRESHOLD, top_k=TOP_K)
        self.mode = mode

        # load existing weights
        if weights_path is not None:
            self._model.load_weights(weights_path, by_name=True)
    
    def __call__(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        return self._model(inputs)
