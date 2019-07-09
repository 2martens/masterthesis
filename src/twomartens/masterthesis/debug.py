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
    save_ssd_train_images(images, labels, output_path):
        saves the first batch of SSD train images with overlaid ground truth bounding boxes
"""
import os

import math
import numpy as np
from matplotlib import pyplot
from PIL import Image

from twomartens.masterthesis import config
from twomartens.masterthesis.ssd_keras.eval_utils import coco_utils


def save_ssd_train_images(images: np.ndarray, labels: np.ndarray, output_path: str) -> None:
    annotation_file_train = f"{config.get_property('Paths.coco')}/annotations/instances_train2014.json"
    _, _, _, classes_to_names = coco_utils.get_coco_category_maps(annotation_file_train)
    colors = pyplot.cm.hsv(np.linspace(0, 1, 81)).tolist()
    os.makedirs(output_path, exist_ok=True)
    
    nr_images = len(images)
    nr_digits = math.ceil(math.log10(nr_images))
    image_size = config.get_property("Parameters.ssd_image_size")
    
    for i, train_image in enumerate(images):
        instances = labels[i]
        image = Image.fromarray(train_image)
        image.save(f"{output_path}"
                   f"train_image{str(i).zfill(nr_digits)}.png")
        
        figure = pyplot.figure(figsize=(20, 12))
        pyplot.imshow(image)
        
        current_axis = pyplot.gca()
        
        for instance in instances:
            xmin = (instance[-12] + instance[-8]) * image_size
            ymin = (instance[-11] + instance[-7]) * image_size
            xmax = (instance[-10] + instance[-6]) * image_size
            ymax = (instance[-9] + instance[-5]) * image_size
            class_id = np.argmax(instance[:-12], axis=0)
            color = colors[class_id]
            label = f"{classes_to_names[class_id]}"
            current_axis.add_patch(
                pyplot.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=color, fill=False,
                                 linewidth=2))
            current_axis.text(xmin, ymin, label, size='x-large', color='white',
                              bbox={'facecolor': color, 'alpha': 1.0})
        pyplot.savefig(f"{output_path}/bboxes{str(i).zfill(nr_digits)}.png")
        pyplot.close(figure)
