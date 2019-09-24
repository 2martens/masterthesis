# -*- coding: utf-8 -*-

#   Copyright 2018 Timon Brüning, Inga Kempfert, Anne Kunstmann, Jim Martens,
#                  Marius Pierenkemper, Yanneck Reiss
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""
Handle plotting functionality.

Functions:
    save_ssd_train_images(...):
        saves the first batch of SSD train images with overlaid ground truth bounding boxes
    draw_bbox_figure(...):
        draws a bounding box figure and saves it under the image file name under the output path
"""
import functools
import os
from typing import Dict
from typing import Tuple
from typing import Union, Sequence

import math
import numpy as np
from matplotlib import pyplot
from PIL import Image


def save_ssd_train_images(images: Union[np.ndarray, Sequence[str]], labels: np.ndarray,
                          output_path: str, coco_path: str,
                          image_size: int, get_coco_cat_maps_func: callable,
                          custom_string: str = None) -> None:
    """
    Saves given images and labels to given output path.
    
    The images are saved both in a raw version and with bounding boxes printed on them.
    
    Args:
        images: a NumPy array of images or a list of filenames
        labels: a NumPy array of labels
        output_path: path to save the images in
        coco_path: path to the COCO data set
        image_size: size of the resized images
        get_coco_cat_maps_func: callable that returns the COCO category maps for a given annotation file
        custom_string: optional custom string that is prepended to file names
    """
    
    annotation_file_train = f"{coco_path}/annotations/instances_minival2014.json"
    _, _, _, classes_to_names = get_coco_cat_maps_func(annotation_file_train)
    colors = pyplot.cm.hsv(np.linspace(0, 1, 61)).tolist()
    os.makedirs(output_path, exist_ok=True)
    
    nr_images = len(images)
    nr_digits = math.ceil(math.log10(nr_images))
    custom_string = f"{custom_string}_" if custom_string is not None else ""
    
    for i, train_image in enumerate(images):
        instances = labels[i]
        if type(train_image) is str:
            with Image.open(train_image) as _image:
                train_image = np.array(_image, dtype=np.uint8)
        image = Image.fromarray(train_image)
        image.save(f"{output_path}/"
                   f"{custom_string}train_image{str(i).zfill(nr_digits)}.png")
        figure_filename = f"{output_path}/{custom_string}bboxes{str(i).zfill(nr_digits)}.png"
        
        _draw_bbox_image(image=image,
                         filename=figure_filename,
                         draw_func=functools.partial(_draw_bboxes,
                                                     image_size=image_size,
                                                     classes_to_names=classes_to_names),
                         drawables=[
                             (colors, instances)
                         ])


def draw_bbox_figure(image_filename: str, labels: Sequence[np.ndarray],
                     instances: Sequence[Sequence[np.ndarray]],
                     image_size: int,
                     output_path: str, coco_path: str,
                     get_coco_cat_maps_func: callable,
                     suffix: str) -> None:
    """
    Draws a bounding box figure and saves it under the image file name under the output path.
    
    Args:
        image_filename: complete path to image file
        labels: ground truth labels for image
        instances: list of predictions to be compared against each other
        image_size: size of the resized images
        output_path: path to save the images in
        coco_path: path to the COCO data set
        get_coco_cat_maps_func: callable that returns the COCO category maps for a given annotation file
        suffix: suffix of the saved figure file
    """
    annotation_file_train = f"{coco_path}/annotations/instances_minival2014.json"
    _, _, _, classes_to_names = get_coco_cat_maps_func(annotation_file_train)

    colors = pyplot.cm.hsv(np.linspace(0, 1, len(instances))).tolist()
    os.makedirs(output_path, exist_ok=True)
    with Image.open(image_filename) as _image:
        np_image = np.array(_image, dtype=np.uint8)
    image = Image.fromarray(np_image)
    figure_filename = f"{output_path}/{os.path.basename(image_filename)}_bboxes_{suffix}.png"
    drawables = [(colors[i], _instances) for i, _instances in enumerate(instances)]
    drawables.append(([35/255, 45/255, 215/255, 1.0], labels))
    _draw_bbox_image(image=image,
                     filename=figure_filename,
                     draw_func=functools.partial(
                         _draw_bboxes,
                         image_size=image_size,
                         classes_to_names=classes_to_names
                     ),
                     drawables=drawables)
    

def _draw_bbox_image(image: Image,
                     filename: str,
                     draw_func: callable,
                     drawables: Sequence[Tuple[Union[Sequence, float], Sequence[np.ndarray]]]):
    figure = pyplot.figure(figsize=(6.4, 4.8))
    pyplot.imshow(image)
    
    current_axis = pyplot.gca()
    for colors, instances in drawables:
        draw_func(instances=instances,
                  axis=current_axis,
                  colors=colors)
    
    pyplot.savefig(filename, bbox_inches='tight')
    pyplot.close(figure)


def _draw_bboxes(instances: Sequence[np.ndarray], axis: pyplot.Axes,
                 image_size: int,
                 colors: Sequence,
                 classes_to_names: Dict[int, str]) -> None:
    for instance in instances:
        if not len(instance):
            continue
        else:
            class_id, confidence, xmin, ymin, xmax, ymax = _get_bbox_info(instance, image_size)
        
        if class_id == 0:
            continue
        
        if len(np.shape(colors)) == 1:
            color = colors
        else:
            color = colors[class_id]
        label = f"{classes_to_names[class_id]} - {confidence}" \
            if confidence is not None \
            else f"{classes_to_names[class_id]}"
        axis.add_patch(
            pyplot.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=color, fill=False,
                             linewidth=2))
        axis.text(xmin, ymin, label, size='x-large', color='white',
                  bbox={'facecolor': color, 'alpha': 1.0})


def _get_bbox_info(instance: np.ndarray, image_size: int) -> Tuple[int, Union[float, None], float, float, float, float]:
    if len(instance) == 5:  # ground truth
        class_id = int(instance[0])
        confidence = None
        xmin = instance[1]
        ymin = instance[2]
        xmax = instance[3]
        ymax = instance[4]
    elif len(instance) == 7:  # predictions
        class_id = int(instance[0])
        confidence = instance[1]
        xmin = instance[3]
        ymin = instance[4]
        xmax = instance[5]
        ymax = instance[6]
    elif len(instance) == 6:  # predictions using Caffe method
        class_id = int(instance[0])
        confidence = instance[1]
        xmin = instance[2]
        ymin = instance[3]
        xmax = instance[4]
        ymax = instance[5]
    else:
        instance = np.copy(instance)
        class_id = np.argmax(instance[:-12], axis=0)
        confidence = np.amax(instance[:-12], axis=0)
        instance[-12:-8] *= instance[-4:]  # multiply with variances
        instance[[-11, -9]] *= np.expand_dims(instance[-5] - instance[-7], axis=-1)
        instance[[-12, -10]] *= np.expand_dims(instance[-6] - instance[-8], axis=-1)
        instance[-12:-8] += instance[-8:-4]
        instance[-12:-8] *= image_size
        
        xmin = instance[-12]
        ymin = instance[-11]
        xmax = instance[-10]
        ymax = instance[-9]
    
    return class_id, confidence, xmin, ymin, xmax, ymax
