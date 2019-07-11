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
    N_CLASSES: number of known classes (without background)
    
Functions:
    compile_model(...): compiles an SSD model
    get_loss_func(...): returns the SSD loss function
    get_model(...): returns correct SSD model and corresponding predictor sizes
    predict(...): runs trained SSD/DropoutSSD on a given data set
    train(...): trains the SSD/DropoutSSD on a given data set
"""
import os
import pickle

from typing import List, Sequence, Tuple
from typing import Optional

import numpy as np
import tensorflow as tf

from twomartens.masterthesis.ssd_keras.bounding_box_utils import bounding_box_utils
from twomartens.masterthesis.ssd_keras.data_generator import object_detection_2d_misc_utils
from twomartens.masterthesis.ssd_keras.keras_loss_function import keras_ssd_loss
from twomartens.masterthesis.ssd_keras.ssd_encoder_decoder import ssd_output_decoder

K = tf.keras.backend
tfe = tf.contrib.eager

N_CLASSES = 80


def get_model(use_dropout: bool,
              dropout_model: callable, vanilla_model: callable,
              image_size: int, nr_classes: int, mode: str,
              iou_threshold: float, dropout_rate: float, top_k: int,
              pre_trained_weights_file: Optional[str] = None) -> Tuple[tf.keras.models.Model, np.ndarray]:
    """
    Returns the correct SSD model and the corresponding predictor sizes.
    
    Args:
        use_dropout: True if dropout variant should be used, False otherwise
        dropout_model: function to build dropout SSD model
        vanilla_model: function to build vanilla SSD model
        image_size: size of the resized images
        nr_classes: number of classes
        mode: one of "training", "inference", "inference_fast"
        iou_threshold: all boxes with higher iou to local maximum box are suppressed
        dropout_rate: rate for dropout layers (only applies if dropout is used)
        top_k: number of highest scoring predictions kept for each batch item
        pre_trained_weights_file: path to h5 file with pre-trained weights

    Returns:
        SSD model, predictor_sizes
    """
    
    image_size = (image_size, image_size, 3)
    scales = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
    if use_dropout:
        model, predictor_sizes = dropout_model(
            image_size=image_size,
            n_classes=nr_classes,
            mode=mode,
            iou_threshold=iou_threshold,
            dropout_rate=dropout_rate,
            top_k=top_k,
            scales=scales,
            return_predictor_sizes=True,
            coords="corners"
        )
    else:
        model, predictor_sizes = vanilla_model(
            image_size=image_size,
            n_classes=nr_classes,
            mode=mode,
            iou_threshold=iou_threshold,
            top_k=top_k,
            scales=scales,
            return_predictor_sizes=True,
            coords="corners"
        )
    
    if mode == "training":
        # set non-classifier layers to non-trainable
        classifier_names = ['conv4_3_norm_mbox_conf',
                            'fc7_mbox_conf',
                            'conv6_2_mbox_conf',
                            'conv7_2_mbox_conf',
                            'conv8_2_mbox_conf',
                            'conv9_2_mbox_conf']
        for layer in model.layers:
            if layer.name not in classifier_names:
                layer.trainable = False
    
    if pre_trained_weights_file is not None:
        model.load_weights(pre_trained_weights_file, by_name=True)
    
    return model, predictor_sizes


def get_loss_func() -> callable:
    return keras_ssd_loss.SSDLoss().compute_loss


def compile_model(model: tf.keras.models.Model, learning_rate: float, loss_func: callable) -> None:
    """
    Compiles an SSD model.
    
    Args:
        model: SSD model
        learning_rate: the learning rate
        loss_func: loss function to minimize
    """
    learning_rate_var = K.variable(learning_rate)
    
    # compile the model
    model.compile(
        optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate_var,
                                         beta1=0.9, beta2=0.999),
        loss=loss_func,
        metrics=[
            "categorical_accuracy"
        ]
    )


def predict(generator: callable,
            steps_per_epoch: int,
            ssd_model: tf.keras.models.Model,
            use_dropout: bool,
            forward_passes_per_image: int,
            image_size: int,
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
    image_size = (image_size, image_size)
    
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


def train(train_generator: callable,
          steps_per_epoch_train: int,
          val_generator: callable,
          steps_per_epoch_val: int,
          ssd_model: tf.keras.models.Model,
          weights_prefix: str,
          iteration: int,
          initial_epoch: int,
          nr_epochs: int,
          tensorboard_callback: Optional[tf.keras.callbacks.TensorBoard]) -> tf.keras.callbacks.History:
    """
    Trains the SSD on the given data set using Keras functionality.
    
    Args:
        train_generator: generator of training data
        steps_per_epoch_train: number of batches per training epoch
        val_generator: generator of validation data
        steps_per_epoch_val: number of batches per validation epoch
        ssd_model: compiled SSD model
        weights_prefix: prefix for weights directory
        iteration: identifier for current training run
        initial_epoch: the epoch to start training in
        nr_epochs: number of epochs to train
        tensorboard_callback: initialised TensorBoard callback
    """
    
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
    
    history = ssd_model.fit_generator(generator=train_generator,
                                      epochs=nr_epochs,
                                      steps_per_epoch=steps_per_epoch_train,
                                      validation_data=val_generator,
                                      validation_steps=steps_per_epoch_val,
                                      callbacks=callbacks,
                                      initial_epoch=initial_epoch)
    
    ssd_model.save(f"{checkpoint_dir}/ssd300.h5")
    ssd_model.save_weights(f"{checkpoint_dir}/ssd300_weights.h5")
    
    return history


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
