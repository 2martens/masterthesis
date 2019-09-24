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
from typing import Union

import math
import numpy as np
import tensorflow as tf
from attributedict.collections import AttributeDict

from twomartens.masterthesis import config
from twomartens.masterthesis import plotting
from twomartens.masterthesis.ssd_keras.bounding_box_utils import bounding_box_utils
from twomartens.masterthesis.ssd_keras.data_generator import object_detection_2d_misc_utils
from twomartens.masterthesis.ssd_keras.eval_utils import coco_utils
from twomartens.masterthesis.ssd_keras.keras_loss_function import keras_ssd_loss
from twomartens.masterthesis.ssd_keras.ssd_encoder_decoder import ssd_output_decoder

K = tf.keras.backend
tfe = tf.contrib.eager


def get_model(use_bayesian: bool,
              bayesian_model: callable, vanilla_model: callable,
              conf_obj: config.Config,
              mode: str,
              pre_trained_weights_file: Optional[str] = None) -> Tuple[tf.keras.models.Model, np.ndarray]:
    """
    Returns the correct SSD model and the corresponding predictor sizes.
    
    Args:
        use_bayesian: True if Bayesian variant should be used, False otherwise
        bayesian_model: function to build Bayesian SSD model
        vanilla_model: function to build vanilla SSD model
        conf_obj: configuration object
        mode: one of "training", "inference", "inference_fast"
        pre_trained_weights_file: path to h5 file with pre-trained weights

    Returns:
        SSD model, predictor_sizes
    """
    
    image_size = (conf_obj.parameters.ssd_image_size, conf_obj.parameters.ssd_image_size, 3)
    scales = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
    if use_bayesian:
        model, predictor_sizes = bayesian_model(
            image_size=image_size,
            n_classes=conf_obj.parameters.nr_classes,
            mode=mode,
            iou_threshold=conf_obj.parameters.ssd_iou_threshold,
            dropout_rate=conf_obj.parameters.ssd_dropout_rate,
            top_k=conf_obj.parameters.ssd_top_k,
            scales=scales,
            return_predictor_sizes=True,
            coords="corners",
            use_dropout=conf_obj.parameters.ssd_use_dropout
        )
    else:
        model, predictor_sizes = vanilla_model(
            image_size=image_size,
            n_classes=conf_obj.parameters.nr_classes,
            mode=mode,
            iou_threshold=conf_obj.parameters.ssd_iou_threshold,
            top_k=conf_obj.parameters.ssd_top_k,
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
    """Returns loss function for SSD."""
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
            conf_obj: config.Config,
            steps_per_epoch: int,
            use_bayesian: bool,
            nr_digits: int,
            output_path: str) -> None:
    """
    Run trained SSD on the given data set.
    
    The prediction results are saved to the output path.
    
    Args:
        generator: generator of test data
        model: compiled and trained Keras model
        conf_obj: configuration object
        steps_per_epoch: number of batches per epoch
        use_bayesian: if True, multiple forward passes and observations will be used
        nr_digits: number of digits needed to print largest batch number
        output_path: the path in which the results should be saved
    """
    output_file, label_output_file = _predict_prepare_paths(output_path, use_bayesian)
    
    _predict_loop(
        generator=generator,
        use_bayesian=use_bayesian,
        conf_obj=conf_obj,
        steps_per_epoch=steps_per_epoch,
        callables=AttributeDict({
            "dropout_step": functools.partial(
                _predict_dropout_step,
                model=model,
                batch_size=conf_obj.parameters.batch_size,
                forward_passes_per_image=conf_obj.parameters.ssd_forward_passes_per_image
            ),
            "vanilla_step": functools.partial(_predict_vanilla_step, model=model),
            "save_images": functools.partial(
                _predict_save_images,
                save_images=plotting.save_ssd_train_images,
                get_coco_cat_maps_func=coco_utils.get_coco_category_maps,
                output_path=output_path,
                conf_obj=conf_obj
            ),
            "decode_func": functools.partial(
                _decode_predictions,
                decode_func=ssd_output_decoder.decode_detections,
                image_size=conf_obj.parameters.ssd_image_size,
                confidence_threshold=conf_obj.parameters.ssd_confidence_threshold,
                iou_threshold=conf_obj.parameters.ssd_iou_threshold,
                top_k=conf_obj.parameters.ssd_top_k
            ),
            "decode_func_dropout": functools.partial(
                _decode_predictions_dropout,
                decode_func=ssd_output_decoder.decode_detections_dropout,
                image_size=conf_obj.parameters.ssd_image_size,
                confidence_threshold=conf_obj.parameters.ssd_confidence_threshold,
            ),
            "apply_entropy_threshold_func": functools.partial(
                _apply_entropy_filtering,
                confidence_threshold=conf_obj.parameters.ssd_confidence_threshold,
                nr_classes=conf_obj.parameters.nr_classes,
                iou_threshold=conf_obj.parameters.ssd_iou_threshold,
                use_nms=conf_obj.parameters.ssd_use_nms
            ),
            "apply_top_k_func": functools.partial(
                _apply_top_k,
                top_k=conf_obj.parameters.ssd_top_k
            ),
            "get_observations_func": _get_observations,
            "transform_func": functools.partial(
                _transform_predictions,
                inverse_transform_func=object_detection_2d_misc_utils.apply_inverse_transforms),
            "save_func": functools.partial(
                _save_predictions,
                output_file=output_file,
                label_output_file=label_output_file,
                nr_digits=nr_digits
            )
        })
    )


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


def _predict_loop(generator: Generator,
                  use_bayesian: bool,
                  conf_obj: config.Config,
                  steps_per_epoch: int,
                  callables: AttributeDict) -> None:
    
    batch_counter = 0
    saved_images_prediction = False
    saved_images_decoding = False
    if conf_obj.parameters.ssd_use_entropy_threshold:
        nr_steps = math.floor(
            (conf_obj.parameters.ssd_entropy_threshold_max - conf_obj.parameters.ssd_entropy_threshold_min) * 10
        )
        entropy_thresholds = [round(i / 10 + conf_obj.parameters.ssd_entropy_threshold_min, 1) for i in range(nr_steps)]
    else:
        entropy_thresholds = [0]
    
    for inputs, filenames, inverse_transforms, original_labels in generator:
        if use_bayesian:
            predictions = callables.dropout_step(inputs)
        else:
            predictions = callables.vanilla_step(inputs)
        
        if not saved_images_prediction:
            callables.save_images(inputs, predictions, custom_string="after-prediction")
            saved_images_prediction = True
        if use_bayesian:
            decoded_predictions = callables.decode_func_dropout(predictions)
            callables.save_func(decoded_predictions, original_labels, filenames,
                                batch_nr=batch_counter,
                                suffix="_prediction",
                                save_labels=False)
            observations = callables.get_observations_func(decoded_predictions)
            callables.save_func(observations, original_labels, filenames,
                                batch_nr=batch_counter,
                                suffix="_observation",
                                save_labels=False)
        for entropy_threshold in entropy_thresholds:
            if use_bayesian:
                decoded_predictions = callables.apply_entropy_threshold_func(observations,
                                                                             entropy_threshold=entropy_threshold)
                callables.save_func(decoded_predictions, original_labels, filenames,
                                    batch_nr=batch_counter, entropy_threshold=entropy_threshold,
                                    suffix="_entropy",
                                    save_labels=False)
                decoded_predictions = callables.apply_top_k_func(decoded_predictions)
            else:
                decoded_predictions = callables.decode_func(predictions, entropy_threshold=entropy_threshold)
            if not saved_images_decoding:
                custom_string = f"after-decoding-{entropy_threshold}" \
                    if conf_obj.parameters.ssd_use_entropy_threshold else "after-decoding"
                callables.save_images(inputs, decoded_predictions, custom_string=custom_string)
            
            transformed_predictions = callables.transform_func(decoded_predictions,
                                                               inverse_transforms)
            callables.save_func(transformed_predictions, original_labels, filenames,
                                batch_nr=batch_counter, entropy_threshold=entropy_threshold,
                                suffix="_transformed")
        
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
                                confidence_threshold: float,
                                ) -> List[np.ndarray]:
    return decode_func(
        y_pred=predictions,
        img_width=image_size,
        img_height=image_size,
        input_coords="corners",
        confidence_thresh=confidence_threshold
    )


def _apply_entropy_filtering(observations: Sequence[np.ndarray],
                             entropy_threshold: float,
                             confidence_threshold: float,
                             iou_threshold: float,
                             nr_classes: int,
                             use_nms: bool = True) -> List[np.ndarray]:
    final_observations = []
    batch_size = len(observations)
    for i in range(batch_size):
        if not observations[i].size:
            final_observations.append(observations[i])
            continue
        
        filtered_image_observations = observations[i][observations[i][:, -1] < entropy_threshold]
        final_image_observations = []
        for class_id in range(1, nr_classes):
            single_class = filtered_image_observations[:, [class_id, -5, -4, -3, -2]]
            threshold_met = single_class[single_class[:, 0] > confidence_threshold]
            if threshold_met.shape[0] > 0:
                if use_nms:
                    maxima = ssd_output_decoder._greedy_nms(threshold_met, iou_threshold=iou_threshold)
                    maxima_output = np.zeros((maxima.shape[0], maxima.shape[1] + 1))
                else:
                    maxima_output = np.zeros((threshold_met.shape[0], threshold_met.shape[1] + 1))
                maxima_output[:, 0] = class_id
                maxima_output[:, 1:] = maxima if use_nms else threshold_met
                final_image_observations.append(maxima_output)
        if final_image_observations:
            final_image_observations = np.concatenate(final_image_observations, axis=0)
        else:
            final_image_observations = np.array(final_image_observations)
        final_observations.append(final_image_observations)
    
    return final_observations


def _apply_top_k(detections: Sequence[np.ndarray], top_k: float) -> List[np.ndarray]:
    final_detections = []
    batch_size = len(detections)
    data_type = np.dtype([('class_id', np.int32),
                          ('confidence', 'f4'),
                          ('xmin', 'f4'),
                          ('ymin', 'f4'),
                          ('xmax', 'f4'),
                          ('ymax', 'f4')])
    for i in range(batch_size):
        image_detections = detections[i]
        if not image_detections.size:
            final_detections.append(image_detections)
            continue
        
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


def _save_predictions(transformed_predictions: Union[np.ndarray, Sequence[np.ndarray]],
                      original_labels: np.ndarray, filenames: Sequence[str],
                      output_file: str, label_output_file: str,
                      batch_nr: int, nr_digits: int, suffix: str,
                      save_labels: Optional[bool] = True,
                      entropy_threshold: Optional[float] = 0) -> None:
    
    counter_str = str(batch_nr).zfill(nr_digits)
    filename = f"{output_file}{suffix}-{counter_str}"
    filename = f"{filename}-{entropy_threshold}" if entropy_threshold else filename
    label_filename = f"{label_output_file}-{counter_str}.bin"

    with open(filename, "wb") as file, open(label_filename, "wb") as label_file:
        pickle.dump(transformed_predictions, file)
        if save_labels:
            pickle.dump({"labels": original_labels, "filenames": filenames}, label_file)


def _predict_save_images(inputs: np.ndarray, predictions: np.ndarray,
                         save_images: callable,
                         get_coco_cat_maps_func: callable,
                         output_path: str,
                         conf_obj: config.Config,
                         custom_string: str) -> None:
    save_images(inputs, predictions,
                output_path, conf_obj.paths.coco, conf_obj.parameters.ssd_image_size,
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
        if not len(observations[i]):
            continue
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
