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
    LOG_FREQUENCY: number of steps that muss pass before logging happens
    
Classes:
    ``DropoutSSD``: wraps Dropout SSD 300 model
    
    ``SSD``: wraps vanilla SSD 300 model
"""
import os
import time
from typing import Dict
from typing import Optional

import tensorflow as tf
from tensorflow.python.ops import summary_ops_v2

from twomartens.masterthesis.ssd_keras.keras_loss_function import keras_ssd_loss
from twomartens.masterthesis.ssd_keras.models import keras_ssd300
from twomartens.masterthesis.ssd_keras.models import keras_ssd300_dropout

K = tf.keras.backend
tfe = tf.contrib.eager

IMAGE_SIZE = (240, 320, 3)  # TODO check with SceneNet RGB-D
N_CLASSES = 80
DROPOUT_RATE = 0.5
IOU_THRESHOLD = 0.45
TOP_K = 200

LOG_FREQUENCY = 10


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


def train(dataset: tf.data.Dataset,
          iteration: int,
          use_dropout: bool,
          weights_prefix: str,
          weights_path: Optional[str] = None,
          verbose: Optional[bool] = False,
          batch_size: Optional[int] = 128,
          nr_epochs: Optional[int] = 80,
          lr: Optional[float] = 0.002) -> None:
    """
    Trains the SSD on the given data set.
    
    This function provides early stopping and creates checkpoints after every
    epoch as well as after finishing training. When starting
    this function with the same ``iteration`` then the training will try to
    continue where it ended last time by restoring a saved checkpoint.
    The loss values are provided as scalar summaries.
    
    Args:
        dataset: the training data set
        iteration: identifier for current training run
        use_dropout: if True, the DropoutSSD will be used
        weights_prefix: prefix for weights directory
        weights_path: path to the pre-trained SSD weights
        verbose: if True, progress is printed to the standard output
        batch_size: size of each batch
        nr_epochs: number of epochs to train
        lr: initial learning rate
    """
    
    # define checkpointed tensors and variables
    checkpointables = {
        'learning_rate_var': K.variable(lr),
    }
    # model
    if use_dropout:
        checkpointables.update({
            'ssd': DropoutSSD(mode='training', weights_path=weights_path)
        })
    else:
        checkpointables.update({
            'ssd': SSD(mode='training', weights_path=weights_path)
        })
    
    checkpointables.update({
        # optimizer
        'ssd_optimizer': tf.train.AdamOptimizer(learning_rate=checkpointables['learning_rate_var'],
                                                beta1=0.5, beta2=0.999),
        # global step counter
        'global_step': tf.train.get_or_create_global_step(),
        'epoch_var': K.variable(-1, dtype=tf.int64)
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
    
    for epoch in range(nr_epochs - previous_epochs):
        _epoch = epoch + previous_epochs
        outputs = _train_one_epoch(_epoch, dataset, **checkpointables)
        
        if verbose:
            print((
                f"[{_epoch + 1:d}/{nr_epochs:d} - "
                f"train time: {outputs['per_epoch_time']:.2f}, "
                f"SSD loss: {outputs['ssd_loss']:.3f}, "
            ))
        
        # save weights at end of epoch
        checkpoint.save(checkpoint_prefix)
    
    if verbose:
        print("Training finished!... save model weights")
    
    # save trained models
    checkpoint.save(checkpoint_prefix)


def _train_one_epoch(epoch: int,
                     dataset: tf.data.Dataset,
                     ssd: tf.keras.Model,
                     ssd_optimizer: tf.train.Optimizer,
                     global_step: tf.Variable,
                     epoch_var: tf.Variable) -> Dict[str, float]:
    
    with summary_ops_v2.always_record_summaries():
        epoch_var.assign(epoch)
        epoch_start_time = time.time()
        
        # define loss variables
        ssd_loss_avg = tfe.metrics.Mean(name='ssd_loss', dtype=tf.float32)
        
        # go through data set
        for x, y in dataset:
            ssd_train_loss = _train_ssd_step(ssd=ssd,
                                             optimizer=ssd_optimizer,
                                             inputs=x,
                                             ground_truth=y,
                                             global_step=global_step)
            ssd_loss_avg(ssd_train_loss)
            global_step.assign_add(1)
        
        epoch_end_time = time.time()
        per_epoch_time = epoch_end_time - epoch_start_time
        
        # final losses of epoch
        outputs = {
            'ssd_loss': ssd_loss_avg.result(False),
            'per_epoch_time': per_epoch_time,
        }
        
        return outputs


def _train_ssd_step(ssd: tf.keras.Model,
                    optimizer: tf.train.Optimizer,
                    inputs: tf.Tensor,
                    ground_truth: tf.Tensor,
                    global_step: tf.Variable) -> tf.Tensor:
    """
    Trains the SSD model for one step (one batch).
    
    :param ssd: instance of the SSD model
    :param optimizer: instance of chosen optimizer
    :param inputs: inputs from data set
    :param ground_truth: ground truth from data set
    :param global_step: the global step variable
    :return: the calculated loss
    """
    with tf.GradientTape() as tape:
        predictions = ssd(inputs)
        loss = keras_ssd_loss.SSDLoss()
        batch_size = tf.shape(predictions)[0]
        ssd_loss = loss.compute_loss(ground_truth, predictions) / tf.to_float(batch_size)
    
    ssd_grads = tape.gradient(ssd_loss, ssd.trainable_variables)
    if int(global_step % LOG_FREQUENCY) == 0:
        summary_ops_v2.scalar(name='ssd_loss', tensor=ssd_loss, step=global_step)
        
        for grad, variable in zip(ssd_grads, ssd.trainable_variables):
            summary_ops_v2.histogram(name='gradients/' + variable.name, tensor=tf.math.l2_normalize(grad),
                                     step=global_step)
            summary_ops_v2.histogram(name='variables/' + variable.name, tensor=tf.math.l2_normalize(variable),
                                     step=global_step)
    
    optimizer.apply_gradients(zip(ssd_grads, ssd.trainable_variables),
                              global_step=global_step)
    
    return ssd_loss
