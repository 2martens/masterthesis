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

import pickle
from typing import Sequence
from typing import Tuple

import numpy as np
import tensorflow as tf

K = tf.keras.backend


def prepare_training_data(test_fold_id: int,
                          inlier_classes: Sequence[int],
                          total_classes: int,
                          fold_prefix: str = 'data/data_fold_',
                          batch_size: int = 128,
                          folds: int = 5) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Prepares the MNIST training data.
    
    Args:
        test_fold_id: id of test fold
        inlier_classes: list of class ids that are considered inliers
        total_classes: total number of classes
        fold_prefix: the prefix for the fold pickle files (default: 'data/data_fold_')
        batch_size: size of batch (default: 128)
        folds: number of folds (default: 5)
    
    Returns:
        A tuple (train dataset, valid dataset)
    """
    # prepare data
    mnist_train = []
    mnist_valid = []
    
    for i in range(folds):
        if i != test_fold_id:  # exclude testing fold, representing 20% of each class
            with open(f"{fold_prefix}{i:d}.pkl", 'rb') as pkl:
                fold = pickle.load(pkl)
            if len(mnist_valid) == 0:  # single out one fold, comprising 20% of each class
                mnist_valid = fold
            else:  # form train set from remaining folds, comprising 60% of each class
                mnist_train += fold
    
    outlier_classes = []
    for i in range(total_classes):
        if i not in inlier_classes:
            outlier_classes.append(i)
    
    # keep only train classes
    mnist_train = [x for x in mnist_train if x[0] in inlier_classes]
    
    def _list_of_pairs_to_numpy(list_of_pairs: Sequence[Tuple[int, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts a list of pairs to a numpy array.

        Args:
            list_of_pairs: list of pairs
        
        Returns:
            tuple (feature array, label array)
        """
        return np.asarray([x[1] for x in list_of_pairs], np.float32), np.asarray([x[0] for x in list_of_pairs], np.int)
    
    mnist_train_x, mnist_train_y = _list_of_pairs_to_numpy(mnist_train)
    mnist_valid_x, mnist_valid_y = _list_of_pairs_to_numpy(mnist_valid)
    
    # get dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((mnist_train_x, mnist_train_y))
    train_dataset = train_dataset.shuffle(mnist_train_x.shape[0]).batch(batch_size,
                                                                        drop_remainder=True).map(_normalize)
    valid_dataset = tf.data.Dataset.from_tensor_slices((mnist_valid_x, mnist_valid_y))
    valid_dataset = valid_dataset.shuffle(mnist_valid_x.shape[0]).batch(batch_size,
                                                                        drop_remainder=True).map(_normalize)
    
    return train_dataset, valid_dataset


def _normalize(feature: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Normalizes a tensor from a 0-255 range to a 0-1 range and adds one dimension.
    
    :param feature: tensor to be normalized
    :param label: label tensor
    :return: normalized tensor
    """
    return K.expand_dims(tf.divide(feature, 255.0)), label
