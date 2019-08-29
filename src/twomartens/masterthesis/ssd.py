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

Functions:
    compile_model(...): compiles an SSD model
    get_loss_func(...): returns the SSD loss function
    get_model(...): returns correct SSD model and corresponding predictor sizes
    predict(...): runs trained SSD/DropoutSSD on a given data set
    train(...): trains the SSD/DropoutSSD on a given data set
"""
import functools
import os
import pickle
from typing import Generator
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

import math
import numpy as np
import tensorflow as tf

from twomartens.masterthesis import debug
from twomartens.masterthesis.ssd_keras.bounding_box_utils import bounding_box_utils
from twomartens.masterthesis.ssd_keras.data_generator import object_detection_2d_misc_utils
from twomartens.masterthesis.ssd_keras.eval_utils import coco_utils
from twomartens.masterthesis.ssd_keras.keras_loss_function import keras_ssd_loss
from twomartens.masterthesis.ssd_keras.ssd_encoder_decoder import ssd_output_decoder

K = tf.keras.backend
tfe = tf.contrib.eager


def get_model(use_dropout: bool,
              dropout_model: callable, vanilla_model: callable,
              image_size: int, nr_classes: int, mode: str,
              dropout_rate: float, top_k: int,
              pre_trained_weights_file: Optional[str] = None,
              iou_threshold: Optional[float] = None) -> Tuple[tf.keras.models.Model, np.ndarray]:
    """
    Returns the correct SSD model and the corresponding predictor sizes.
    
    Args:
        use_dropout: True if dropout variant should be used, False otherwise
        dropout_model: function to build dropout SSD model
        vanilla_model: function to build vanilla SSD model
        image_size: size of the resized images
        nr_classes: number of classes
        mode: one of "training", "inference", "inference_fast"
        dropout_rate: rate for dropout layers (only applies if dropout is used)
        top_k: number of highest scoring predictions kept for each batch item
        pre_trained_weights_file: path to h5 file with pre-trained weights
        iou_threshold: all boxes with higher iou to local maximum box are suppressed
                       (only relevant for inference modes)

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
            model: tf.keras.models.Model,
            steps_per_epoch: int,
            image_size: int,
            batch_size: int,
            forward_passes_per_image: int,
            use_entropy_threshold: bool,
            entropy_threshold_min: float,
            entropy_threshold_max: float,
            confidence_threshold: float,
            iou_threshold: float,
            top_k: int,
            output_path: str,
            coco_path: str,
            use_dropout: bool,
            nr_digits: int) -> None:
    """
    Run trained SSD on the given data set.
    
    The prediction results are saved to the output path.
    
    Args:
        generator: generator of test data
        model: compiled and trained Keras model
        steps_per_epoch: number of batches per epoch
        image_size: size of input images to model
        batch_size: number of items in every batch
        forward_passes_per_image: specifies number of forward passes per image
            used by DropoutSSD
        use_entropy_threshold: if True entropy thresholding is applied
        entropy_threshold_min: specifies the minimum threshold for the entropy
        entropy_threshold_max: specifies the maximum threshold for the entropy
        confidence_threshold: minimum confidence required for box to count as positive
        iou_threshold: all boxes with iou overlap larger than threshold to local maximum box
                       will be suppressed
        top_k: a maximum of top_k boxes remain after NMS
        output_path: the path in which the results should be saved
        coco_path: the path to the COCO data set
        use_dropout: if True, multiple forward passes and observations will be used
        nr_digits: number of digits needed to print largest batch number
    """
    output_file, label_output_file = _predict_prepare_paths(output_path, use_dropout)
    
    _predict_loop(generator, use_dropout, steps_per_epoch,
                  dropout_step=functools.partial(_predict_dropout_step,
                                                 model=model,
                                                 batch_size=batch_size,
                                                 forward_passes_per_image=forward_passes_per_image),
                  vanilla_step=functools.partial(_predict_vanilla_step, model=model),
                  save_images=functools.partial(_predict_save_images,
                                                save_images=debug.save_ssd_train_images,
                                                get_coco_cat_maps_func=coco_utils.get_coco_category_maps,
                                                output_path=output_path,
                                                coco_path=coco_path,
                                                image_size=image_size),
                  decode_func=functools.partial(
                      _decode_predictions,
                      decode_func=ssd_output_decoder.decode_detections,
                      image_size=image_size,
                      confidence_threshold=confidence_threshold,
                      iou_threshold=iou_threshold,
                      top_k=top_k
                  ),
                  decode_func_dropout=functools.partial(
                      _decode_predictions_dropout,
                      decode_func=ssd_output_decoder.decode_detections_dropout,
                      image_size=image_size,
                      confidence_threshold=confidence_threshold,
                  ),
                  apply_entropy_threshold_func=_apply_entropy_threshold,
                  apply_top_k_func=functools.partial(
                      _apply_top_k,
                      top_k=top_k
                  ),
                  get_observations_func=_get_observations,
                  transform_func=functools.partial(
                      _transform_predictions,
                      inverse_transform_func=object_detection_2d_misc_utils.apply_inverse_transforms),
                  save_func=functools.partial(_save_predictions,
                                              output_file=output_file,
                                              label_output_file=label_output_file,
                                              nr_digits=nr_digits),
                  use_entropy_threshold=use_entropy_threshold,
                  entropy_threshold_min=entropy_threshold_min,
                  entropy_threshold_max=entropy_threshold_max)


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


def _predict_prepare_paths(output_path: str, use_dropout: bool) -> Tuple[str, str]:
    filename = "ssd_predictions"
    label_filename = "ssd_labels"
    if use_dropout:
        filename = f"dropout-{filename}"
    
    output_file = os.path.join(output_path, filename)
    label_output_file = os.path.join(output_path, label_filename)
    
    return output_file, label_output_file


def _predict_loop(generator: Generator, use_dropout: bool, steps_per_epoch: int,
                  dropout_step: callable, vanilla_step: callable,
                  save_images: callable, decode_func: callable,
                  decode_func_dropout: callable, get_observations_func: callable,
                  apply_entropy_threshold_func: callable, apply_top_k_func: callable,
                  transform_func: callable, save_func: callable,
                  use_entropy_threshold: bool, entropy_threshold_min: float,
                  entropy_threshold_max: float) -> None:
    
    batch_counter = 0
    saved_images_prediction = False
    saved_images_decoding = False
    if use_entropy_threshold:
        nr_steps = math.floor((entropy_threshold_max - entropy_threshold_min) * 10)
        entropy_thresholds = [round(i / 10 + entropy_threshold_min, 1) for i in range(nr_steps)]
    else:
        entropy_thresholds = [0]
    
    for inputs, filenames, inverse_transforms, original_labels in generator:
        if use_dropout:
            predictions = dropout_step(inputs)
        else:
            predictions = vanilla_step(inputs)
        
        if not saved_images_prediction:
            save_images(inputs, predictions, custom_string="after-prediction")
            saved_images_prediction = True
        if use_dropout:
            decoded_predictions = decode_func_dropout(predictions)
            observations = get_observations_func(decoded_predictions)
        for entropy_threshold in entropy_thresholds:
            if use_dropout:
                decoded_predictions = apply_entropy_threshold_func(observations, entropy_threshold=entropy_threshold)
                decoded_predictions = apply_top_k_func(decoded_predictions)
            else:
                decoded_predictions = decode_func(predictions, entropy_threshold=entropy_threshold)
            if not saved_images_decoding:
                custom_string = f"after-decoding-{entropy_threshold}" if use_entropy_threshold else "after-decoding"
                save_images(inputs, decoded_predictions, custom_string=custom_string)
            
            transformed_predictions = transform_func(decoded_predictions,
                                                     inverse_transforms)
            save_func(transformed_predictions, original_labels, filenames,
                      batch_nr=batch_counter, entropy_threshold=entropy_threshold)
        
        if not saved_images_decoding:
            saved_images_decoding = True
        
        batch_counter += 1
        
        if batch_counter == steps_per_epoch:
            break


def _predict_dropout_step(inputs: np.ndarray, model: tf.keras.models.Model,
                          batch_size: int, forward_passes_per_image: int) -> np.ndarray:
    
    detections = np.zeros((batch_size, 8732 * forward_passes_per_image, 73))
    
    for forward_pass in range(forward_passes_per_image):
        predictions = model.predict_on_batch(inputs)
        
        for i in range(batch_size):
            detections[i][forward_pass * 8732:forward_pass * 8732 + 8732] = predictions[i]
    
    return detections


def _predict_vanilla_step(inputs: np.ndarray, model: tf.keras.models.Model) -> np.ndarray:
    return np.asarray(model.predict_on_batch(inputs))


def _decode_predictions(predictions: np.ndarray,
                        decode_func: callable,
                        image_size: int,
                        entropy_threshold: float,
                        confidence_threshold: float,
                        iou_threshold: float,
                        top_k: int) -> np.ndarray:
    return decode_func(
        y_pred=predictions,
        img_width=image_size,
        img_height=image_size,
        input_coords="corners",
        entropy_thresh=entropy_threshold,
        confidence_thresh=confidence_threshold,
        iou_threshold=iou_threshold,
        top_k=top_k
    )


def _decode_predictions_dropout(predictions: np.ndarray,
                                decode_func: callable,
                                image_size: int,
                                # entropy_threshold: float,
                                confidence_threshold: float,
                                # iou_threshold: float,
                                # top_k: int
                                ) -> List[np.ndarray]:
    return decode_func(
        y_pred=predictions,
        img_width=image_size,
        img_height=image_size,
        input_coords="corners",
        confidence_thresh=confidence_threshold
    )


def _apply_entropy_threshold(observations: Sequence[np.ndarray], entropy_threshold: float) -> List[np.ndarray]:
    final_observations = []
    batch_size = len(observations)
    for i in range(batch_size):
        if not observations[i].size:
            final_observations.append(observations[i])
            continue
        
        filtered_image_observations = observations[i][observations[i][:, -1] < entropy_threshold]
        final_image_observations = np.copy(filtered_image_observations[:, -8:-1])
        final_image_observations[:, 0] = np.argmax(filtered_image_observations[:, :-5], axis=-1)
        final_image_observations[:, 1] = np.amax(filtered_image_observations[:, :-5], axis=-1)
        final_image_observations[:, 2] = filtered_image_observations[:, -1]
        final_observations.append(final_image_observations)
    
    return final_observations


def _apply_top_k(detections: Sequence[np.ndarray], top_k: float) -> List[np.ndarray]:
    final_detections = []
    batch_size = len(detections)
    data_type = np.dtype([('class_id', np.int32),
                          ('confidence', 'f4'),
                          ('entropy', 'f4'),
                          ('xmin', 'f4'),
                          ('ymin', 'f4'),
                          ('xmax', 'f4'),
                          ('ymax', 'f4')])
    for i in range(batch_size):
        image_detections = detections[i]
        image_detections_structured = np.core.records.fromarrays(image_detections.transpose(),
                                                                 dtype=data_type)
        descending_indices = np.argsort(-image_detections_structured['confidence'])
        image_detections_sorted = image_detections[descending_indices]
        if image_detections_sorted.shape[0] > top_k:
            top_k_indices = np.argpartition(image_detections_sorted[:, 1],
                                            kth=image_detections_sorted.shape[0] - top_k,
                                            axis=0)[image_detections_sorted.shape[0] - top_k:]
            final_detections.append(image_detections_sorted[top_k_indices])
        else:
            final_detections.append(image_detections_sorted)
    
    return final_detections


def _transform_predictions(decoded_predictions: np.ndarray, inverse_transforms: Sequence[np.ndarray],
                           inverse_transform_func: callable) -> np.ndarray:
    
    return inverse_transform_func(decoded_predictions, inverse_transforms)


def _save_predictions(transformed_predictions: np.ndarray, original_labels: np.ndarray, filenames: Sequence[str],
                      output_file: str, label_output_file: str,
                      batch_nr: int, nr_digits: int, entropy_threshold: float) -> None:
    
    counter_str = str(batch_nr).zfill(nr_digits)
    filename = f"{output_file}-{counter_str}"
    filename = f"{filename}-{entropy_threshold}" if entropy_threshold else filename
    label_filename = f"{label_output_file}-{counter_str}.bin"

    with open(filename, "wb") as file, open(label_filename, "wb") as label_file:
        pickle.dump(transformed_predictions, file)
        pickle.dump({"labels": original_labels, "filenames": filenames}, label_file)


def _predict_save_images(inputs: np.ndarray, predictions: np.ndarray,
                         save_images: callable,
                         get_coco_cat_maps_func: callable,
                         output_path: str, coco_path: str,
                         image_size: int, custom_string: str) -> None:
    save_images(inputs, predictions,
                output_path, coco_path, image_size,
                get_coco_cat_maps_func, custom_string)
    

def _get_observations(detections: Sequence[np.ndarray]) -> List[np.ndarray]:
    batch_size = len(detections)
    observations = [[] for _ in range(batch_size)]
    final_observations = []
    
    # iterate over images
    for i in range(batch_size):
        detections_image = np.asarray(detections[i])
        if not detections_image.size:
            final_observations.append(detections_image)
            continue
        
        overlaps = bounding_box_utils.iou(detections_image[:, -5:-1],
                                          detections_image[:, -5:-1],
                                          mode="outer_product",
                                          border_pixels="include")
        image_observations = []
        used_boxes = None
        for j in range(overlaps.shape[0]):
            # check if box is already in existing observation
            if used_boxes is not None and j in used_boxes:
                continue
            
            box_overlaps = overlaps[j]
            overlap_detections = np.nonzero(box_overlaps >= 0.95)
            if not len(overlap_detections[0]):
                continue
            observation_set = np.unique(overlap_detections, axis=0)
            for k in overlap_detections[0]:
                # check if box was already removed from observation, then skip
                if k not in observation_set:
                    continue
                
                # check if other found detections are also overlapping with this
                # detection
                second_overlaps = overlaps[k]
                second_detections = np.unique(np.nonzero(second_overlaps >= 0.95), axis=0)
                difference = np.setdiff1d(observation_set, second_detections, assume_unique=True)
                observation_set = np.setdiff1d(observation_set, difference)
            
            if used_boxes is None:
                used_boxes = observation_set
            else:
                used_boxes = np.unique(np.concatenate([used_boxes, observation_set],
                                                      axis=0), axis=0)
            image_observations.append(observation_set)
        
        for observation in image_observations:
            observation_detections = detections_image[observation]
            # average over class probabilities
            observation_mean = np.mean(observation_detections, axis=0)
            observations[i].append(observation_mean)
        
        final_observations.append(np.asarray(observations[i]))
        final_observations[i][:, -1] = -np.sum(final_observations[i][:, :-5] * np.log(final_observations[i][:, :-5]),
                                               axis=-1)
        
    return final_observations


def _set_difference(first_array: np.ndarray, second_array: np.ndarray) -> np.ndarray:
    """
    Removes all elements from first_array that are present in second_array.
    
    Args:
        first_array: the first array
        second_array: the second array
    
    Returns:
        set difference between first_array and second_array
    """
    max2 = second_array.max(axis=0)
    max1 = first_array.max(axis=0)
    dims = np.maximum(max2,
                      max1) + 1
    rmi2 = np.ravel_multi_index(second_array.T, dims)
    rmi1 = np.ravel_multi_index(first_array.T, dims)
    in1d = np.in1d(rmi2, rmi1)
    return second_array[~in1d]
