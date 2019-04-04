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
Functionality to load COCO data into Tensorflow data sets.

Functions:
    load_coco(...): loads the COCO data into a Tensorflow data set
"""
from typing import Tuple

import tensorflow as tf
from pycocotools import coco


def load_coco(data_path: str, data_type: str, num_epochs: int, batch_size: int = 32) -> tf.data.Dataset:
    """
    Loads the COCO data and returns a data set.
    
    Args:
        data_path: path to the COCO data set
        data_type: type of the COCO data (e.g. 'val2014')
        num_epochs: number of epochs
        batch_size: batch size (default: 32)
    Returns:
        Tensorflow data set
    """
    annotation_file = f"{data_path}/annotations/instances_{data_type}.json"
    coco_interface = coco.COCO(annotation_file)
    img_ids = coco_interface.getImgIds()  # return all image IDs
    images = coco_interface.loadImgs(img_ids)  # load all images
    annotation_ids = coco_interface.getAnnIds(img_ids)
    annotations = coco_interface.loadAnns(annotation_ids)  # load all image annotations
    file_names = [image['file_name'] for image in images]
    cat_ids = [annotation['category_id'] for annotation in annotations]
    length_dataset = len(file_names)

    def _load_image(image_data: Tuple[str, int]):
        path, label = image_data
        image = tf.read_file(f"{data_path}/images/{path}")
        image = tf.image.decode_image(image, channels=3)
    
        return image, label
    
    # build image data set
    path_dataset = tf.data.Dataset.from_tensor_slices(file_names)
    label_dataset = tf.data.Dataset.from_tensor_slices(cat_ids)
    dataset = tf.data.Dataset.zip((path_dataset, label_dataset))
    dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=length_dataset, count=num_epochs))
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.map(_load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    return dataset
