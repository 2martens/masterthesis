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
    
Functions:
    predict(...): runs trained SSD/DropoutSSD on a given data set
    train(...): trains the SSD/DropoutSSD on a given data set
"""
import math
import os
import pickle
import time
from typing import Dict, List, Sequence, Union, Tuple
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import summary_ops_v2

from twomartens.masterthesis.ssd_keras.bounding_box_utils import bounding_box_utils
from twomartens.masterthesis.ssd_keras.data_generator import object_detection_2d_misc_utils
from twomartens.masterthesis.ssd_keras.keras_loss_function import keras_ssd_loss
from twomartens.masterthesis.ssd_keras.models import keras_ssd300
from twomartens.masterthesis.ssd_keras.models import keras_ssd300_dropout
from twomartens.masterthesis.ssd_keras.ssd_encoder_decoder import ssd_output_decoder
from twomartens.masterthesis.ssd_keras.ssd_encoder_decoder import ssd_input_encoder

K = tf.keras.backend
tfe = tf.contrib.eager

IMAGE_SIZE = (300, 300, 3)
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
    
    Attributes:
        mode: one of training, inference, and inference_fast
        predictor_sizes: sizes of predictor layers
        model: Keras SSD model
    """
    
    def __init__(self, mode: str, weights_path: Optional[str] = None) -> None:
        self.model, self.predictor_sizes = \
            keras_ssd300.ssd_300(image_size=IMAGE_SIZE, n_classes=N_CLASSES,
                                 mode=mode, iou_threshold=IOU_THRESHOLD, top_k=TOP_K,
                                 scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
                                 return_predictor_sizes=True)  # type: tf.keras.models.Model, np.ndarray
        self.mode = mode  # type: str
        
        # load existing weights
        if weights_path is not None:
            self.model.load_weights(weights_path, by_name=True)
        
        if mode == "training":
            # set non-classifier layers to non-trainable
            classifier_names = ['conv4_3_norm_mbox_conf',
                                'fc7_mbox_conf',
                                'conv6_2_mbox_conf',
                                'conv7_2_mbox_conf',
                                'conv8_2_mbox_conf',
                                'conv9_2_mbox_conf']
            for layer in self.model.layers:
                if layer.name not in classifier_names:
                    layer.trainable = False
    
    def __call__(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        return self.model(inputs)


class DropoutSSD:
    """
    Wraps Dropout SSD 300 model.
    
    Args:
        mode: one of training, inference, and inference_fast
        weights_path: path to trained weights
    
    Attributes:
        mode: one of training, inference, and inference_fast
        predictor_sizes: sizes of predictor layers
        model: Keras SSD model
    """
    
    def __init__(self, mode: str, weights_path: Optional[str] = None) -> None:
        self.model, self.predictor_sizes = \
            keras_ssd300_dropout.ssd_300_dropout(image_size=IMAGE_SIZE,
                                                 n_classes=N_CLASSES,
                                                 dropout_rate=DROPOUT_RATE, mode=mode,
                                                 iou_threshold=IOU_THRESHOLD,
                                                 top_k=TOP_K,
                                                 scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
                                                 return_predictor_sizes=True)  # type: tf.keras.models.Model, np.ndarray
        self.mode = mode  # type: str

        # load existing weights
        if weights_path is not None:
            self.model.load_weights(weights_path, by_name=True)

        if mode == "training":
            # set non-classifier layers to non-trainable
            classifier_names = ['conv4_3_norm_mbox_conf',
                                'fc7_mbox_conf',
                                'conv6_2_mbox_conf',
                                'conv7_2_mbox_conf',
                                'conv8_2_mbox_conf',
                                'conv9_2_mbox_conf']
            for layer in self.model.layers:
                if layer.name not in classifier_names:
                    layer.trainable = False
    
    def __call__(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        return self.model(inputs)
    

def predict_keras(generator: callable,
                  steps_per_epoch: int,
                  ssd_model: tf.keras.models.Model,
                  use_dropout: bool,
                  forward_passes_per_image: int,
                  image_size: Tuple[int, int],
                  output_path: str,
                  nr_digits: int) -> None:
    """
    Run trained SSD on the given data set.
    
    The prediction results are saved to the output path.
    
    Args:
        generator: generator of test data
        steps_per_epoch: number of batches per epoch
        ssd_model: compiled and trained Keras model
        use_dropout: if True, multiple forward passes and observations will be used
        forward_passes_per_image: specifies number of forward passes per image
            used by DropoutSSD
        image_size: size of input images to model
        output_path: the path in which the results should be saved
        nr_digits: number of digits needed to print largest batch number
    """
    # prepare filename
    filename = 'ssd_predictions'
    label_filename = 'ssd_labels'
    if use_dropout:
        filename = f"dropout-{filename}"
    output_file = os.path.join(output_path, filename)
    label_output_file = os.path.join(output_path, label_filename)
    
    batch_counter = 0
    for x, filenames, inverse_transforms, original_labels in generator:
        if use_dropout:
            detections = None
            batch_size = None
            for _ in range(forward_passes_per_image):
                predictions = ssd_model.predict_on_batch(x)
                if batch_size is None:
                    batch_size = predictions.shape[0]
                if detections is None:
                    detections = [[] for _ in range(batch_size)]
                
                for i in range(batch_size):
                    batch_item = predictions[i]
                    detections[i].extend(batch_item)
                
            # do observation stuff
            predictions = np.asarray(_get_observations(detections))
        else:
            predictions = np.asarray(ssd_model.predict_on_batch(x))
            print(predictions[:, :, [-2, -1]])
        
        decoded_predictions_batch = ssd_output_decoder.decode_detections_fast(
            y_pred=predictions,
            img_width=image_size[0],
            img_height=image_size[1],
        )
        transformed_predictions_batch = object_detection_2d_misc_utils.apply_inverse_transforms(
            decoded_predictions_batch, inverse_transforms
        )
        
        # save prediction results to prevent memory issues
        counter_str = str(batch_counter).zfill(nr_digits)
        filename = f"{output_file}-{counter_str}.bin"
        label_filename = f"{label_output_file}-{counter_str}.bin"
        
        with open(filename, 'wb') as file, open(label_filename, 'wb') as label_file:
            pickle.dump(transformed_predictions_batch, file)
            pickle.dump({'labels': original_labels, 'filenames': filenames}, label_file)
        
        batch_counter += 1
        # we only do one epoch for prediction
        if batch_counter == steps_per_epoch:
            break


def predict(dataset: tf.data.Dataset,
            use_dropout: bool,
            output_path: str,
            weights_path: Optional[str] = None,
            checkpoint_path: Optional[str] = None,
            verbose: Optional[bool] = False,
            forward_passes_per_image: Optional[int] = 42,
            nr_digits: Optional[int] = None) -> None:
    """
    Run trained SSD on the given data set.
    
    Either the weights path or the checkpoint path must be given. This prevents
    a scenario where an untrained network is used to predict.
    
    The prediction results are saved to the output path.
    
    Args:
        dataset: the testing data set
        use_dropout: if True, DropoutSSD will be used
        output_path: the path in which the results should be saved
        weights_path: the path to the trained Keras weights (h5 file)
        checkpoint_path: the path to the stored checkpoints (Tensorflow checkpoints)
        verbose: if True, progress is printed to the standard output
        forward_passes_per_image: specifies number of forward passes per image
            used by DropoutSSD
        nr_digits: number of digits needed to print largest batch number
    """
    if weights_path is None and checkpoint_path is None:
        raise ValueError("Either 'weights_path' or 'checkpoint_path' must be given.")

    # model
    if use_dropout:
        ssd = DropoutSSD(mode='training', weights_path=weights_path)
    else:
        ssd = SSD(mode='inference_fast', weights_path=weights_path)
    
    checkpointables = {
        'ssd': ssd.model,
        'learning_rate_var': K.variable(0),
    }

    checkpointables.update({
        # optimizer
        'ssd_optimizer': tf.train.AdamOptimizer(learning_rate=checkpointables['learning_rate_var'],
                                                beta1=0.5, beta2=0.999),
        # global step counter
        'global_step':   tf.train.get_or_create_global_step(),
        'epoch_var':     K.variable(-1, dtype=tf.int64)
    })

    if checkpoint_path is not None:
        # checkpoint
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
        checkpoint = tf.train.Checkpoint(**checkpointables)
        checkpoint.restore(latest_checkpoint)

    outputs = _predict_one_epoch(dataset, use_dropout, output_path, forward_passes_per_image,
                                 nr_digits, checkpointables['ssd'])
    
    if verbose:
        print((
            f"predict time: {outputs['per_epoch_time']:.2f}, "
        ))
        print("Prediction finished!... save outputs")


def _predict_one_epoch(dataset: tf.data.Dataset,
                       use_dropout: bool,
                       output_path: str,
                       forward_passes_per_image: int,
                       nr_digits: int,
                       ssd: tf.keras.Model) -> Dict[str, float]:
    
    epoch_start_time = time.time()
   
    # prepare filename
    filename = 'ssd_predictions'
    label_filename = 'ssd_labels'
    if use_dropout:
        filename = f"dropout-{filename}"
    output_file = os.path.join(output_path, filename)
    label_output_file = os.path.join(output_path, label_filename)
    
    # go through the data set
    counter = 0
    
    for inputs, labels in dataset:
        if use_dropout:
            detections = None
            batch_size = None
            for _ in range(forward_passes_per_image):
                result = np.array(ssd(inputs))
                if batch_size is None:
                    batch_size = result.shape[0]
                if detections is None:
                    detections = [[] for _ in range(batch_size)]
                
                for i in range(batch_size):
                    batch_item = result[i]
                    detections[i].extend(batch_item)
            
            observations = np.asarray(_get_observations(detections))
            del detections
            
            observations = ssd_output_decoder.decode_detections_fast(observations,
                                                                     img_height=IMAGE_SIZE[0],
                                                                     img_width=IMAGE_SIZE[1])
            result_transformed = []
            for i in range(batch_size):
                # apply inverse transformations to predicted bounding box coordinates
                x_reverse = labels[i, 0, 5]
                y_reverse = labels[i, 0, 6]
                filtered = observations[i]
                filtered[:, 2] *= x_reverse
                filtered[:, 4] *= x_reverse
                filtered[:, 3] *= y_reverse
                filtered[:, 5] *= y_reverse
                result_transformed.append(filtered)
            decoded_predictions_batch = result_transformed
        else:
            result = np.array(ssd(inputs))
            result_filtered = []
            # iterate over result of images
            for i in range(result.shape[0]):
                # apply inverse transformations to predicted bounding box coordinates
                # filter out dummy all-zero results
                x_reverse = labels[i, 0, 5]
                y_reverse = labels[i, 0, 6]
                filtered = result[i][result[i, :, 0] != 0]
                filtered[:, 2] *= x_reverse
                filtered[:, 4] *= x_reverse
                filtered[:, 3] *= y_reverse
                filtered[:, 5] *= y_reverse
                result_filtered.append(filtered)
            decoded_predictions_batch = result_filtered
        
        # save predictions batch-wise to prevent memory problems
        if nr_digits is not None:
            counter_str = str(counter).zfill(nr_digits)
            filename = f"{output_file}-{counter_str}.bin"
            label_filename = f"{label_output_file}-{counter_str}.bin"
        else:
            filename = f"{output_file}-{counter:d}.bin"
            label_filename = f"{label_output_file}-{counter:d}.bin"
        
        with open(filename, 'wb') as file, open(label_filename, 'wb') as label_file:
            pickle.dump(decoded_predictions_batch, file)
            pickle.dump(labels, label_file)
        
        counter += 1
    
    epoch_end_time = time.time()
    per_epoch_time = epoch_end_time - epoch_start_time

    # outputs for epoch
    outputs = {
        'per_epoch_time': per_epoch_time,
    }

    return outputs


def _get_observations(detections: Sequence[Sequence[np.ndarray]]) -> List[List[np.ndarray]]:
    batch_size = len(detections)
    observations = [[] for _ in range(batch_size)]
    print(f"batch size: {batch_size}")
    
    # iterate over images
    for i in range(batch_size):
        detections_image = np.asarray(detections[i])
        overlaps = bounding_box_utils.iou(detections_image[:, -12:-8],
                                          detections_image[:, -12:-8],
                                          mode="outer_product",
                                          border_pixels="include")
        image_observations = []
        used_boxes = set()
        for j in range(overlaps.shape[0]):
            # check if box is already in existing observation
            if j in used_boxes:
                continue
            
            box_overlaps = overlaps[j]
            overlap_detections = np.nonzero(box_overlaps >= 0.95)
            observation_set = set(overlap_detections)
            for k in overlap_detections:
                # check if box was already removed from observation, then skip
                if k not in observation_set:
                    continue
                
                # check if other found detections are also overlapping with this
                # detection
                second_overlaps = overlaps[k]
                second_detections = set(np.nonzero(second_overlaps >= 0.95))
                difference = observation_set - second_detections
                observation_set = observation_set - difference
            
            used_boxes.update(observation_set)
            image_observations.append(observation_set)
        
        for observation in image_observations:
            observation_detections = detections_image[np.asarray(list(observation))]
            # average over class probabilities
            observation_mean = np.mean(observation_detections, axis=0)
            observations[i].append(observation_mean)
    
    return observations


def train_keras(train_generator: callable,
                steps_per_epoch_train: int,
                val_generator: callable,
                steps_per_epoch_val: int,
                ssd_model: Union[SSD, DropoutSSD],
                weights_prefix: str,
                iteration: int,
                initial_epoch: int,
                nr_epochs: int,
                lr: float,
                tensorboard_callback: Optional[tf.keras.callbacks.TensorBoard]) -> tf.keras.callbacks.History:
    """
    Trains the SSD on the given data set using Keras functionality.
    
    Args:
        train_generator: generator of training data
        steps_per_epoch_train: number of batches per training epoch
        val_generator: generator of validation data
        steps_per_epoch_val: number of batches per validation epoch
        ssd_model: wrapper of SSD model
        weights_prefix: prefix for weights directory
        iteration: identifier for current training run
        initial_epoch: the epoch to start training in
        nr_epochs: number of epochs to train
        lr: initial learning rate
        tensorboard_callback: initialised TensorBoard callback
    """
    
    # set up variables
    learning_rate_var = K.variable(lr)
    ssd_loss = keras_ssd_loss.SSDLoss()
    
    # compile the model
    ssd_model.model.compile(
        optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate_var,
                                         beta1=0.9, beta2=0.999),
        loss=ssd_loss.compute_loss,
        metrics=[
            "categorical_accuracy"
        ]
    )

    checkpoint_dir = os.path.join(weights_prefix, str(iteration))
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{checkpoint_dir}/ssd300-{{epoch:02d}}_loss-{{loss:.4f}}_val_loss-{{val_loss:.4f}}.h5",
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=False
        ),
        tf.keras.callbacks.TerminateOnNaN(),
        # tf.keras.callbacks.EarlyStopping(patience=2, min_delta=0.001, monitor="val_loss")
    ]
    if tensorboard_callback is not None:
        callbacks.append(tensorboard_callback)
    
    history = ssd_model.model.fit_generator(generator=train_generator,
                                            epochs=nr_epochs,
                                            steps_per_epoch=steps_per_epoch_train,
                                            validation_data=val_generator,
                                            validation_steps=steps_per_epoch_val,
                                            callbacks=callbacks,
                                            initial_epoch=initial_epoch)
    
    ssd_model.model.save(f"{checkpoint_dir}/ssd300.h5")
    ssd_model.model.save_weights(f"{checkpoint_dir}/ssd300_weights.h5")
    
    return history


def train(dataset: tf.data.Dataset,
          iteration: int,
          use_dropout: bool,
          length_dataset: int,
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
        length_dataset: specifies number of images in data set
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
        ssd = DropoutSSD(mode='training', weights_path=weights_path)
    else:
        ssd = SSD(mode='training', weights_path=weights_path)
    
    checkpointables.update({
        'ssd': ssd.model
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
    # update model inside SSD object with version from checkpoint
    ssd.model = checkpointables['ssd']

    # input encoder
    input_encoder = ssd_input_encoder.SSDInputEncoder(IMAGE_SIZE[0], IMAGE_SIZE[1],
                                                      N_CLASSES, ssd.predictor_sizes,
                                                      steps=[8, 16, 32, 64, 100, 300],
                                                      aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                                                               [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                                               [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                                               [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                                               [1.0, 2.0, 0.5],
                                                                               [1.0, 2.0, 0.5]])
    
    with summary_ops_v2.always_record_summaries():
        summary_ops_v2.scalar(name='learning_rate', tensor=checkpointables['learning_rate_var'],
                              step=checkpointables['global_step'])
        
    nr_batches_per_epoch = int(math.ceil(length_dataset / float(batch_size)))
    
    _train_epochs(nr_batches_per_epoch, nr_epochs, dataset, input_encoder,
                  checkpoint, checkpoint_prefix, verbose=verbose, **checkpointables)
    
    if verbose:
        print("Training finished!... save model weights")
    
    # save trained models
    checkpoint.save(checkpoint_prefix)


def _train_epochs(nr_batches_per_epoch: int,
                  nr_epochs: int,
                  dataset: tf.data.Dataset,
                  input_encoder: ssd_input_encoder.SSDInputEncoder,
                  checkpoint: tf.train.Checkpoint,
                  checkpoint_prefix: str,
                  ssd: tf.keras.Model,
                  ssd_optimizer: tf.train.Optimizer,
                  global_step: tf.Variable,
                  epoch_var: tf.Variable,
                  learning_rate_var: tf.Variable,
                  verbose: bool) -> None:
    
    with summary_ops_v2.always_record_summaries():
        epoch = 0
        batch_counter = 0
        epoch_var.assign(epoch)
        
        # go through data set
        for x, y in dataset:
            if batch_counter == 0:
                # epoch starts
                epoch_start_time = time.time()
                ssd_loss_avg = tfe.metrics.Mean(name='ssd_loss', dtype=tf.float32)
                if verbose:
                    print((
                        f"epoch: {epoch + 1:d}"
                    ))
            
            labels = []
            for i in range(y.shape[0]):
                image_labels = np.asarray(y[i])
                image_labels = image_labels[image_labels[:, 0] != -1]
                labels.append(image_labels)
            encoded_ground_truth = input_encoder(labels)
            ssd_train_loss = _train_ssd_step(ssd=ssd,
                                             optimizer=ssd_optimizer,
                                             inputs=x,
                                             ground_truth=encoded_ground_truth,
                                             global_step=global_step)
            ssd_loss_avg(ssd_train_loss)
            global_step.assign_add(1)

            batch_counter += 1

            if batch_counter == nr_batches_per_epoch:
                # one epoch is over
                epoch_end_time = time.time()
                per_epoch_time = epoch_end_time - epoch_start_time

                # final losses of epoch
                outputs = {
                    'ssd_loss':       ssd_loss_avg.result(False),
                    'per_epoch_time': per_epoch_time,
                }

                if verbose:
                    print((
                        f"[{epoch + 1:d}/{nr_epochs:d}] - "
                        f"train time: {outputs['per_epoch_time']:.2f}, "
                        f"SSD loss: {outputs['ssd_loss']:.3f}, "
                        f"batch_counter: {batch_counter:d}, "
                        f"nr_batches_per_epoch: {nr_batches_per_epoch:d}"
                    ))

                # save weights at end of epoch
                checkpoint.save(checkpoint_prefix)
                
                epoch += 1
                
                batch_counter = 0


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
