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

"""aae.util.py: contains utility functions"""
import math
from typing import Sequence, Tuple, Union

import numpy as np
import tensorflow as tf

k = tf.keras.backend


def save_image(tensor: tf.Tensor, filename: str, **kwargs) -> None:
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        filename (string): name of image
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    grid = make_grid(tensor, **kwargs)
    min_pixel_value = 0
    max_pixel_value = 255
    grid *= max_pixel_value
    grid = tf.clip_by_value(grid, min_pixel_value, max_pixel_value)
    grid = tf.cast(grid, tf.uint8)
    ndarr = grid.cpu().numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)


def make_grid(tensor: Union[tf.Tensor, Sequence[tf.Tensor]], nrow: int = 8, padding: int = 2,
              normalize: bool = False, range_value: Tuple[float, float] = None,
              scale_each: bool = False, pad_value: float = 0.0) -> tf.Tensor:
    """Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The Final grid size is (B / nrow, nrow). Default is 8.
        padding (int, optional): amount of padding. Default is 2.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by subtracting the minimum and dividing by the maximum pixel value.
        range_value (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If True, scale each image in the batch of
            images separately rather than the (min, max) over all images.
        pad_value (float, optional): Value for the padded pixels.

    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_

    """
    if not (tf.contrib.framework.is_tensor(tensor) or
            (isinstance(tensor, list) and all(tf.contrib.framework.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))
    
    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = k.stack(tensor, axis=0)
    tensor_shape = tf.shape(tensor).numpy()
    tensor_rank = tf.rank(tensor).numpy()
    
    if tensor_rank == 2:  # single image H x W
        tensor = k.reshape(tensor, (tensor_shape[0], tensor_shape[1], 1))
    if tensor_rank == 3:  # single image
        if tensor_shape[2] == 1:  # if single-channel, convert to 3-channel
            tensor = k.concatenate((tensor, tensor, tensor), axis=2)
        tensor = k.reshape(tensor, (1, tensor_shape[0], tensor_shape[1], tensor_shape[2]))
    
    if tensor_rank == 4 and tensor_shape[3] == 1:  # single-channel images
        tensor = k.concatenate((tensor, tensor, tensor), axis=3)
    
    if normalize is True:
        if range_value is not None:
            assert isinstance(range_value, tuple), \
                "range_value has to be a tuple (min, max) if specified. min and max are numbers"
        
        def norm_ip(img: tf.Tensor, min_v: float, max_v: float) -> tf.Tensor:
            """
            Internal function to clip given tensor to given min and max values.
            :param img: tensor to be clipped
            :param min_v: min value
            :param max_v: max value
            :return: clipped tensor
            """
            img = tf.clip_by_value(img, min_v, max_v)
            img = tf.add(img, -min_v)
            return tf.divide(img, max_v - min_v + 1e-5)
        
        def norm_range(t: tf.Tensor, range_v: Tuple[float, float] = None) -> tf.Tensor:
            """
            Internal function to normalize a tensor to a given range.
            :param t: tensor to be normalized
            :param range_v: tuple with (min, max) range values
            :return: normalized tensor
            """
            if range_v is not None:
                return norm_ip(t, range_v[0], range_v[1])
            else:
                return norm_ip(t, float(k.min(t)), float(k.max(t)))
        
        if scale_each is True:
            updated_tensors = []
            for t in tensor:  # loop over mini-batch dimension
                updated_tensors.append(norm_range(t, range_value))
            tensor = k.constant(np.array(updated_tensors))
        else:
            tensor = norm_range(tensor, range_value)
    
    if tensor_shape[0] == 1:
        return tf.squeeze(tensor)
    
    # make the mini-batch of images into a grid
    nmaps = tensor_shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor_shape[1] + padding), int(tensor_shape[2] + padding)
    grid = tf.fill((height * ymaps + padding, width * xmaps + padding, 3), pad_value).numpy()
    tensor_numpy = tensor.numpy()
    i = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if i >= nmaps:
                break
            start_height = y * height + padding
            start_width = x * width + padding
            np.copyto(grid[start_height: start_height + height - padding,
                      start_width:start_width + width - padding], tensor_numpy[i, :, :, :])
            i = i + 1
    return k.constant(grid)
