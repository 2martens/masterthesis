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
Handle debug functionality.

Functions:
    save_ssd_train_images(...):
        saves the first batch of SSD train images with overlaid ground truth bounding boxes
"""
import os

import math
import numpy as np
from matplotlib import pyplot
from PIL import Image


def save_ssd_train_images(images: np.ndarray, labels: np.ndarray,
                          output_path: str, coco_path: str,
                          image_size: int, get_coco_cat_maps_func: callable,
                          custom_string: str = None) -> None:
    """
    Saves given images and labels to given output path.
    
    The images are saved both in a raw version and with bounding boxes printed on them.
    
    Args:
        images: a NumPy array of images
        labels: a NumPy array of labels
        output_path: path to save the images in
        coco_path: path to the COCO data set
        image_size: size of the resized images
        get_coco_cat_maps_func: callable that returns the COCO category maps for a given annotation file
        custom_string: optional custom string that is prepended to file names
    """
    
    annotation_file_train = f"{coco_path}/annotations/instances_train2014.json"
    _, _, _, classes_to_names = get_coco_cat_maps_func(annotation_file_train)
    colors = pyplot.cm.hsv(np.linspace(0, 1, 81)).tolist()
    os.makedirs(output_path, exist_ok=True)
    
    nr_images = len(images)
    nr_digits = math.ceil(math.log10(nr_images))
    custom_string = f"{custom_string}_" if custom_string is not None else ""
    
    for i, train_image in enumerate(images):
        instances = labels[i]
        image = Image.fromarray(train_image)
        image.save(f"{output_path}/"
                   f"{custom_string}train_image{str(i).zfill(nr_digits)}.png")
        
        figure = pyplot.figure(figsize=(20, 12))
        pyplot.imshow(image)
        
        current_axis = pyplot.gca()
        
        for instance in instances:
            if len(instance) == 5:
                class_id = instance[0]
                xmin = instance[1]
                ymin = instance[2]
                xmax = instance[3]
                ymax = instance[4]
            else:
                class_id = np.argmax(instance[:-12], axis=0)
                xmin = instance[-12] + instance[-8]
                ymin = instance[-11] + instance[-7]
                xmax = instance[-10] + instance[-6]
                ymax = instance[-9] + instance[-5]
            
            if class_id == 0:
                continue
            
            xmin *= image_size
            ymin *= image_size
            xmax *= image_size
            ymax *= image_size
            
            print(class_id)
            color = colors[class_id]
            label = f"{classes_to_names[class_id]}"
            current_axis.add_patch(
                pyplot.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=color, fill=False,
                                 linewidth=2))
            current_axis.text(xmin, ymin, label, size='x-large', color='white',
                              bbox={'facecolor': color, 'alpha': 1.0})
        pyplot.savefig(f"{output_path}/{custom_string}bboxes{str(i).zfill(nr_digits)}.png")
        pyplot.close(figure)
