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
Functionality to load data into Tensorflow data sets.

Functions:
    load_coco(...): loads the COCO data into a Tensorflow data set
    load_scenenet(...): loads the SceneNet RGB-D data into a Tensorflow data set
"""
from typing import List
from typing import Sequence

import tensorflow as tf
from pycocotools import coco


def load_coco(data_path: str, category: int,
              num_epochs: int, batch_size: int = 32,
              resized_shape: Sequence[int] = (256, 256)) -> tf.data.Dataset:
    """
    Loads the COCO trainval35k data and returns a data set.
    
    Args:
        data_path: path to the COCO data set
        category: id of the inlying class
        num_epochs: number of epochs
        batch_size: batch size (default: 32)
        resized_shape: shape of images after resizing them (default: (300,300))
    Returns:
        Tensorflow data set
    """
    annotation_file_train = f"{data_path}/annotations/instances_train2014.json"
    annotation_file_val = f"{data_path}/annotations/instances_valminusminival2014.json"
    # load training images
    coco_train = coco.COCO(annotation_file_train)
    img_ids = coco_train.getImgIds(catIds=[category])  # return all image IDs belonging to given category
    images = coco_train.loadImgs(img_ids)  # load all images
    annotation_ids = coco_train.getAnnIds(img_ids)
    annotations = coco_train.loadAnns(annotation_ids)  # load all image annotations
    file_names = [f"{data_path}/images/train2014/{image['file_name']}" for image in images]
    bboxes = [annotation['bbox'] for annotation in annotations]
    
    # load validation images
    coco_val = coco.COCO(annotation_file_val)
    img_ids = coco_val.getImgIds(catIds=[category])  # return all image IDs belonging to given category
    images = coco_val.loadImgs(img_ids)  # load all images
    annotation_ids = coco_val.getAnnIds(img_ids)
    annotations = coco_val.loadAnns(annotation_ids)  # load all image annotations
    file_names_val = [f"{data_path}/images/val2014/{image['file_name']}" for image in images]
    bboxes_val = [annotation['bbox'] for annotation in annotations]
    
    file_names.extend(file_names_val)
    bboxes.extend(bboxes_val)
    
    length_dataset = len(file_names)

    def _load_image(paths: Sequence[str], labels: Sequence[Sequence[float]]):
        _images = tf.map_fn(lambda path: tf.read_file(path), paths)
    
        def _get_images(image_data: Sequence[tf.Tensor]) -> List[tf.Tensor]:
            image = tf.image.decode_image(image_data[0], channels=3, dtype=tf.float32)
            image_shape = tf.shape(image)
            image = tf.reshape(image, [image_shape[0], image_shape[1], 3])
            label = image_data[1]
            x1, x2, y1, y2 = tf.cast(tf.round(label[0]), dtype=tf.int32), \
                             tf.cast(tf.round(label[0] + label[2]), dtype=tf.int32), \
                             tf.cast(tf.round(label[1]), dtype=tf.int32), \
                             tf.cast(tf.round(label[1] + label[3]), dtype=tf.int32)
            # image_cut = image[x1:x2 + 1, y1:y2 + 1]
            image_resized = tf.image.resize_image_with_pad(image, resized_shape[0], resized_shape[1])
        
            return [image_resized, label]
    
        processed = tf.map_fn(_get_images, [_images, labels], dtype=[tf.float32, tf.float32])
        processed_images = processed[0]
        processed_images = tf.reshape(processed_images, [-1, resized_shape[0], resized_shape[1], 3])
    
        return processed_images
    
    # build image data set
    path_dataset = tf.data.Dataset.from_tensor_slices(file_names)
    label_dataset = tf.data.Dataset.from_tensor_slices(bboxes)
    dataset = tf.data.Dataset.zip((path_dataset, label_dataset))
    dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=length_dataset, count=num_epochs))
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.map(_load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    return dataset


def load_scenenet(data_path: str, num_epochs: int, batch_size: int = 32) -> tf.data.Dataset:
    """
    Loads the SceneNet RGB-D data and returns a data set.
    
    Args:
        data_path: path to the SceneNet RGB-D data set
        num_epochs: number of epochs
        batch_size: batch size
    Returns:
        Tensorflow data set
    """
    pass
