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
    annotation_ids = coco_train.getAnnIds(img_ids, catIds=[category])
    annotations = coco_train.loadAnns(annotation_ids)  # load all image annotations
    file_names = {image['id']: f"{data_path}/train2014/{image['file_name']}" for image in images}
    
    # load validation images
    coco_val = coco.COCO(annotation_file_val)
    img_ids = coco_val.getImgIds(catIds=[category])  # return all image IDs belonging to given category
    images_val = coco_val.loadImgs(img_ids)  # load all images
    annotation_ids = coco_val.getAnnIds(img_ids, catIds=[category])
    annotations_val = coco_val.loadAnns(annotation_ids)  # load all image annotations
    file_names_val = {image['id']: f"{data_path}/val2014/{image['file_name']}" for image in images_val}

    images.extend(images_val)
    annotations.extend(annotations_val)
    ids_to_images = {image['id']: image for image in images}

    checked_file_names = []
    checked_bboxes = []
    for annotation in annotations:
        img_id = annotation['image_id']
        image = ids_to_images[img_id]
        file_name = file_names[img_id] if img_id in file_names else file_names_val[img_id]
        bbox = annotation['bbox']
        target_height = round(bbox[3])
        target_width = round(bbox[2])
        image_width, image_height = image['width'], image['height']
        y1 = round(bbox[1])
        x1 = round(bbox[0])
        y2 = round(bbox[1] + bbox[3])
        x2 = round(bbox[0] + bbox[2])
        if target_width <= 0 or target_height <= 0:
            continue
        if x2 <= 0 or y2 <= 0:
            continue
        if x1 < 0 or y1 < 0:
            continue
        if x2 + 1 - x1 <= 0 or y2 + 1 - y1 <= 0:
            continue
        if image_width < x2:
            target_width = image_width - x1
        if image_height < y2:
            target_height = image_height - y1
        if target_width <= 0:
            continue
        if target_height <= 0:
            continue
        bbox[2] = target_width
        bbox[3] = target_height
    
        checked_file_names.append(file_name)
        checked_bboxes.append(bbox)

    length_dataset = len(checked_file_names)

    def _load_image(paths: Sequence[str], labels: Sequence[Sequence[float]]):
        _images = tf.map_fn(lambda path: tf.read_file(path), paths)
    
        def _get_images(image_data: Sequence[tf.Tensor]) -> List[tf.Tensor]:
            image = tf.image.decode_image(image_data[0], channels=3, dtype=tf.float32)
            image_shape = tf.shape(image)
            image = tf.reshape(image, [image_shape[0], image_shape[1], 3])
            label = image_data[1]
            image_cut = tf.image.crop_to_bounding_box(image, tf.cast(tf.floor(label[1]), dtype=tf.int32),
                                                      tf.cast(tf.floor(label[0]), dtype=tf.int32),
                                                      tf.cast(tf.floor(label[3]), dtype=tf.int32),
                                                      tf.cast(tf.floor(label[2]), dtype=tf.int32))
            image_resized = tf.image.resize_image_with_pad(image_cut, resized_shape[0], resized_shape[1])
        
            return [image_resized, label]
    
        processed = tf.map_fn(_get_images, [_images, labels], dtype=[tf.float32, tf.float32])
        processed_images = processed[0]
        processed_images = tf.reshape(processed_images, [-1, resized_shape[0], resized_shape[1], 3])
    
        return processed_images
    
    # build image data set
    path_dataset = tf.data.Dataset.from_tensor_slices(checked_file_names)
    label_dataset = tf.data.Dataset.from_tensor_slices(checked_bboxes)
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
