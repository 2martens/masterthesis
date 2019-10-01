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
    load_coco_train(...): loads the COCO training data into a Tensorflow data set
    load_coco_val(...): loads the COCO validation data into a Tensorflow data set
    load_coco_val_ssd(...): loads the COCO validation data using the ssd_keras data pipeline
    load_scenenet_data(...): loads the SceneNet RGB-D data into a Tensorflow data set
    prepare_scenenet_data(...): prepares the SceneNet RGB-D data and returns it in Python format
    group_bboxes_to_images(...): groups bounding boxes to images
    clean_dataset(...): cleans a COCO data set and returns cleaned version
"""
from typing import Callable
from typing import Generator
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np
import scipy
import tensorflow as tf
import tqdm
from scipy import ndimage

from twomartens.masterthesis.ssd_keras.data_generator import data_augmentation_chain_original_ssd
from twomartens.masterthesis.ssd_keras.data_generator import object_detection_2d_data_generator
from twomartens.masterthesis.ssd_keras.data_generator import object_detection_2d_geometric_ops
from twomartens.masterthesis.ssd_keras.data_generator import object_detection_2d_photometric_ops
from twomartens.masterthesis.ssd_keras.ssd_encoder_decoder import ssd_input_encoder


def load_coco_train(data_path: str, category: int,
                    num_epochs: int, batch_size: int = 32,
                    resized_shape: Sequence[int] = (256, 256)) -> tf.data.Dataset:
    """
    Loads the COCO trainval35k data and returns a data set.
    
    Args:
        data_path: path to the COCO data set
        category: id of the inlying class
        num_epochs: number of epochs
        batch_size: batch size (default: 32)
        resized_shape: shape of images after resizing them (default: (256, 256))

    Returns:
        Tensorflow data set
    """
    annotation_file_train = f"{data_path}/annotations/instances_train2014.json"
    annotation_file_val = f"{data_path}/annotations/instances_valminusminival2014.json"
    
    # load training images
    from pycocotools import coco
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
    file_names.update(file_names_val)
    ids_to_images = {image['id']: image for image in images}
    
    checked_file_names, checked_bboxes = clean_dataset(annotations, file_names, ids_to_images)
    length_dataset = len(checked_file_names)
    
    # build image data set
    path_dataset = tf.data.Dataset.from_tensor_slices(checked_file_names)
    label_dataset = tf.data.Dataset.from_tensor_slices(checked_bboxes)
    dataset = tf.data.Dataset.zip((path_dataset, label_dataset))
    dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=length_dataset, count=num_epochs))
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.map(_load_images_callback(resized_shape))
    
    return dataset


def load_coco_val(data_path: str, category: int,
                  num_epochs: int = 1, batch_size: int = 32,
                  resized_shape: Sequence[int] = (256, 256)) -> tf.data.Dataset:
    """
    Loads the COCO minival2014/val2017 data and returns a data set.

    Args:
        data_path: path to the COCO data set
        category: id of the inlying class
        num_epochs: number of epochs (default: 1)
        batch_size: batch size (default: 32)
        resized_shape: shape of images after resizing them (default: (256, 256))

    Returns:
        Tensorflow data set
    """
    annotation_file_minival = f"{data_path}/annotations/instances_minival2014.json"

    from pycocotools import coco
    coco_val = coco.COCO(annotation_file_minival)
    img_ids = coco_val.getImgIds(catIds=[category])  # return all image IDs belonging to given category
    images = coco_val.loadImgs(img_ids)  # load all images
    annotation_ids = coco_val.getAnnIds(img_ids, catIds=[category])
    annotations = coco_val.loadAnns(annotation_ids)  # load all image annotations
    file_names = {image['id']: f"{data_path}/val2014/{image['file_name']}" for image in images}
    ids_to_images = {image['id']: image for image in images}
    
    checked_file_names, checked_bboxes = clean_dataset(annotations, file_names, ids_to_images)
    length_dataset = len(checked_file_names)
    
    # build image data set
    path_dataset = tf.data.Dataset.from_tensor_slices(checked_file_names)
    label_dataset = tf.data.Dataset.from_tensor_slices(checked_bboxes)
    dataset = tf.data.Dataset.zip((path_dataset, label_dataset))
    dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=length_dataset, count=num_epochs))
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.map(_load_images_callback(resized_shape))
    
    return dataset


def clean_dataset(annotations: Sequence[dict], file_names: Mapping[str, str],
                  ids_to_images: Mapping[str, dict]) -> Tuple[List[str], List[List[float]]]:
    """
    Cleans a given data set from problematic cases and returns cleaned version.
    
    Args:
        annotations: list of annotation dictionaries
        file_names: mapping of fileID -> file name
        ids_to_images: mapping of imageID -> image dictionary

    Returns:
        cleaned file names, corresponding clean bounding boxes
    """
    checked_file_names = []
    checked_bboxes = []
    for annotation in annotations:
        img_id = annotation['image_id']
        image = ids_to_images[img_id]
        file_name = file_names[img_id]
        bbox = annotation['bbox']  # type: List[float]
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
        new_bbox = [
            annotation['category_id'],
            x1,
            y1,
            round(bbox[0] + bbox[2]),
            round(bbox[1] + bbox[3])
        ]
        
        checked_file_names.append(file_name)
        checked_bboxes.append(new_bbox)
    
    return checked_file_names, checked_bboxes


def _load_images_callback(resized_shape: Sequence[int]) -> Callable[
        [Sequence[str], Sequence[Sequence[float]]], tf.Tensor]:
    """
    Returns the callback function to load images.
    
    Args:
        resized_shape: shape of resized image (height, width)

    Returns:
        callback function
    """
    
    def _load_images(paths: Sequence[str], labels: Sequence[Sequence[float]]) -> tf.Tensor:
        """
        Callback function to load images.
        
        Args:
            paths: list of file paths
            labels: list of bounding boxes
            
        Returns:
            loaded images
        """
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
    
    return _load_images


def group_bboxes_to_images(file_names: Sequence[str], bboxes: Sequence[Sequence[int]]) -> Tuple[List[str],
                                                                                                List[List[List[int]]]]:
    """
    Groups bounding boxes to images.
    
    Args:
        file_names: list of image file names
        bboxes: list of bounding boxes
    
    Returns:
        tuple of file names and corresponding lists of bounding boxes
    """
    return_labels = {}
    for file_name, bbox in zip(file_names, bboxes):
        if file_name not in return_labels:
            return_labels[file_name] = []
        return_labels[file_name].append(bbox)
    
    return list(return_labels.keys()), list(return_labels.values())


def load_coco_val_ssd(clean_dataset: callable,
                      group_bboxes_to_images: callable,
                      coco_path: str,
                      batch_size: int,
                      image_size: int,
                      training: bool,
                      evaluation: bool,
                      augment: bool,
                      debug: bool,
                      predictor_sizes: Optional[np.ndarray]) -> Tuple[Generator, int, Optional[Generator]]:
    """
    Loads the COCO minival2014/val2017 data and returns a data set.

    Args:
        clean_dataset: function that cleans the data set
        group_bboxes_to_images: function that groups bounding boxes to corresponding file name
        coco_path: path to the COCO data set
        batch_size: batch size
        image_size: size of images after resizing them
        training: True if training data is desired
        evaluation: True if evaluation-ready data is desired
        augment: True if training data should be augmented
        debug: True if a more extensive generator should be added to output
        predictor_sizes: sizes of the predictor layers, can be None for evaluation
        
    Returns:
        coco data set generator
        length of dataset
        generator which offers processed_labels as well (only if debug is True)
    """
    from pycocotools import coco
    
    from twomartens.masterthesis.ssd_keras.eval_utils import coco_utils
    
    annotation_file_minival = f"{coco_path}/annotations/instances_minival2014.json"
    resized_shape = (image_size, image_size)
    cats_to_classes, classes_to_cats, _, _ = coco_utils.get_coco_category_maps(annotation_file_minival)
    
    coco_val = coco.COCO(annotation_file_minival)
    img_ids = coco_val.getImgIds()  # return all image IDs belonging to given category
    images = coco_val.loadImgs(img_ids)  # load all images
    annotation_ids = coco_val.getAnnIds(img_ids)
    annotations = coco_val.loadAnns(annotation_ids)  # load all image annotations
    file_names = {image['id']: f"{coco_path}/val2014/{image['file_name']}" for image in images}
    ids_to_images = {image['id']: image for image in images}
    
    checked_image_paths, checked_bboxes = clean_dataset(annotations, file_names, ids_to_images)
    bboxes_with_converted_cat_ids = []
    for bbox in checked_bboxes:
        bboxes_with_converted_cat_ids.append([
            cats_to_classes[bbox[0]],
            bbox[1],
            bbox[2],
            bbox[3],
            bbox[4]
        ])
    final_image_paths, final_labels = group_bboxes_to_images(checked_image_paths, bboxes_with_converted_cat_ids)

    data_generator = object_detection_2d_data_generator.DataGenerator(
        filenames=final_image_paths,
        labels=final_labels
    )

    shuffle = True if training else False

    if training and augment:
        transformations = [data_augmentation_chain_original_ssd.SSDDataAugmentation(
            img_width=resized_shape[0],
            img_height=resized_shape[1]
        )]
    else:
        transformations = [
            object_detection_2d_photometric_ops.ConvertTo3Channels(),
            object_detection_2d_geometric_ops.Resize(height=resized_shape[0],
                                                     width=resized_shape[1])
        ]

    returns = {'processed_images', 'encoded_labels'}
    returns_debug = {'processed_images', 'encoded_labels', 'processed_labels'}

    if not training and evaluation:
        returns = {
            'processed_images',
            'filenames',
            'inverse_transform',
            'original_labels'}
        label_encoder = None
    else:
        if predictor_sizes is None:
            raise ValueError("predictor_sizes cannot be None for training/validation")
        label_encoder = ssd_input_encoder.SSDInputEncoder(
            img_height=resized_shape[0],
            img_width=resized_shape[1],
            n_classes=len(cats_to_classes),  # 80
            predictor_sizes=predictor_sizes,
            steps=[8, 16, 32, 64, 100, 300],
            coords="corners",
            aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                     [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                     [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                     [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                     [1.0, 2.0, 0.5],
                                     [1.0, 2.0, 0.5]]
        )

    if debug:
        debug_generator = data_generator.generate(
            batch_size=batch_size,
            shuffle=shuffle,
            transformations=transformations,
            label_encoder=label_encoder,
            returns=returns_debug,
            keep_images_without_gt=False
        )
    else:
        debug_generator = None

    length_dataset = data_generator.dataset_size

    generator = data_generator.generate(
        batch_size=batch_size,
        shuffle=shuffle,
        transformations=transformations,
        label_encoder=label_encoder,
        returns=returns,
        keep_images_without_gt=False
    )
    
    return generator, length_dataset, debug_generator


def load_scenenet_data(photo_paths: Sequence[Sequence[str]],
                       instances: Sequence[Sequence[Sequence[dict]]],
                       coco_path: str,
                       batch_size: int,
                       image_size: int,
                       training: bool,
                       evaluation: bool,
                       augment: bool,
                       debug: bool,
                       predictor_sizes: Optional[np.ndarray],
                       nr_trajectories: Optional[int] = None) -> Tuple[Generator, int, Optional[Generator]]:
    """
    Loads the SceneNet RGB-D data and returns a data set.
    
    Args:
        photo_paths: contains a list of image paths per trajectory
        instances: instance data per frame per trajectory
        coco_path: path to the COCO data set
        batch_size: size of every batch
        image_size: size of resized images
        training: True if training data is desired
        evaluation: True if evaluation-ready data is desired
        augment: True if training data should be augmented
        debug: True if a more extensive generator should be added to output
        predictor_sizes: sizes of the predictor layers, can be None for evaluation
        nr_trajectories: number of trajectories to consider

    Returns:
        scenenet data set generator
        length of dataset
        generator which offers processed_labels as well (only if debug is True)
    """
    trajectories = zip(photo_paths, instances)
    final_image_paths = []
    final_labels = []
    resized_shape = (image_size, image_size)

    from twomartens.masterthesis.ssd_keras.eval_utils import coco_utils
    
    annotation_file_train = f"{coco_path}/annotations/instances_train2014.json"
    cats_to_classes, _, _, _ = coco_utils.get_coco_category_maps(annotation_file_train)
    
    for i, trajectory in enumerate(trajectories):
        if nr_trajectories is not None and i >= nr_trajectories:
            break
        
        traj_image_paths, traj_instances = trajectory
        used_images = 0
        for image_path, frame_instances in zip(traj_image_paths, traj_instances):
            labels = []

            if not frame_instances:
                continue  # skip images that do not contain instances
            if evaluation and used_images >= 32:
                continue
            
            for instance in frame_instances:
                bbox = instance['bbox']
                labels.append([
                    float(cats_to_classes[instance['coco_id']]),
                    float(bbox[0]),  # x min
                    float(bbox[1]),  # y min
                    float(bbox[2]),  # x max
                    float(bbox[3]),  # y max
                ])
    
            final_image_paths.append(image_path)
            final_labels.append(labels)
            used_images += 1
    
    data_generator = object_detection_2d_data_generator.DataGenerator(
        filenames=final_image_paths,
        labels=final_labels
    )
    
    shuffle = True if training else False
    
    if training and augment:
        transformations = [data_augmentation_chain_original_ssd.SSDDataAugmentation(
            img_width=resized_shape[0],
            img_height=resized_shape[1]
        )]
    else:
        transformations = [
            object_detection_2d_photometric_ops.ConvertTo3Channels(),
            object_detection_2d_geometric_ops.Resize(height=resized_shape[0],
                                                     width=resized_shape[1])
        ]
        
    returns = {'processed_images', 'encoded_labels'}
    returns_debug = {'processed_images', 'encoded_labels', 'processed_labels'}
    
    if not training and evaluation:
        returns = {
            'processed_images',
            'filenames',
            'inverse_transform',
            'original_labels'}
        label_encoder = None
    else:
        if predictor_sizes is None:
            raise ValueError("predictor_sizes cannot be None for training/validation")
        label_encoder = ssd_input_encoder.SSDInputEncoder(
            img_height=resized_shape[0],
            img_width=resized_shape[1],
            n_classes=len(cats_to_classes),  # 80
            predictor_sizes=predictor_sizes,
            steps=[8, 16, 32, 64, 100, 300],
            coords="corners",
            aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                     [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                     [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                     [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                     [1.0, 2.0, 0.5],
                                     [1.0, 2.0, 0.5]]
        )
    
    generator = data_generator.generate(
        batch_size=batch_size,
        shuffle=shuffle,
        transformations=transformations,
        label_encoder=label_encoder,
        returns=returns,
        keep_images_without_gt=False
    )
    
    if debug:
        debug_generator = data_generator.generate(
            batch_size=batch_size,
            shuffle=shuffle,
            transformations=transformations,
            label_encoder=label_encoder,
            returns=returns_debug,
            keep_images_without_gt=False
        )
    else:
        debug_generator = None
    
    length_dataset = data_generator.dataset_size
    
    return generator, length_dataset, debug_generator


def _load_images_ssd_callback(resized_shape: Sequence[int]) \
        -> Callable[[Sequence[str], Sequence[Sequence[int]]],
                    Tuple[tf.Tensor, Sequence[Sequence[int]]]]:
    """
    Returns the callback function to load images for SSD.

    Args:
        resized_shape: shape of resized image (height, width)

    Returns:
        callback function
    """

    def _load_images_ssd(paths: Sequence[str],
                         labels: Sequence[Sequence[int]]) -> Tuple[tf.Tensor, Sequence[Sequence[int]]]:
        """
        Callback function to load images for SSD.
        
        Args:
            paths: paths to the images
            labels: labels for images
    
        Returns:
            loaded images
        """
        _images = tf.map_fn(lambda path: tf.read_file(path), paths)
    
        def _get_images(data: tf.Tensor) -> Tuple[tf.Tensor, Sequence[int]]:
            image_data, _labels = data
            image = tf.image.decode_image(image_data, channels=3, dtype=tf.float32)
            image_shape = tf.shape(image)
            x_reverse = tf.broadcast_to(
                tf.expand_dims(tf.expand_dims(tf.cast(image_shape[0], dtype=tf.float32) / resized_shape[0],
                               axis=0), axis=0),
                [tf.shape(_labels)[0], 1])
            y_reverse = tf.broadcast_to(
                tf.expand_dims(tf.expand_dims(tf.cast(image_shape[1], dtype=tf.float32) / resized_shape[1],
                               axis=0), axis=0),
                [tf.shape(_labels)[0], 1])
            _labels = tf.concat([_labels, x_reverse, y_reverse], axis=1)
            image = tf.reshape(image, [image_shape[0], image_shape[1], 3])
            image_resized = tf.image.resize_images(image, [resized_shape[0], resized_shape[1]])
            
            return image_resized, _labels
    
        processed = tf.map_fn(_get_images, (_images, labels), dtype=(tf.float32, tf.float32))
        processed_images = tf.reshape(processed[0], [-1, resized_shape[0], resized_shape[1], 3])
    
        return processed_images, processed[1]
    
    return _load_images_ssd


def prepare_scenenet_data(data_path: str, protobuf_path: str) -> Tuple[List[List[str]],
                                                                       List[List[str]],
                                                                       List[List[List[dict]]]]:
    """
    Prepares the SceneNet RGB-D data and returns it in Python format.
    
    Args:
        data_path: path to the SceneNet RGB-D data set
        protobuf_path: path to the SceneNet RGB-D protobuf
    Returns:
        file names photos, file names instances, instances
    """
    from twomartens.masterthesis import definitions
    from twomartens.masterthesis import scenenet_pb2
    
    trajectories = scenenet_pb2.Trajectories()
    with open(protobuf_path, 'rb') as file:
        trajectories.ParseFromString(file.read())

    sorted_trajectories = sorted(trajectories.trajectories, key=lambda k: k.render_path)
    file_names_photos = []
    file_names_instances = []
    instances = []
    for trajectory in tqdm.tqdm(sorted_trajectories, desc="preparing trajectories"):
        path = f"{data_path}/{trajectory.render_path}"
        file_names_photos_traj = []
        file_names_instances_traj = []
        instances_traj = []
        instances_traj_dict = {}
        
        for instance in trajectory.instances:
            instance_type = instance.instance_type
            instance_id = instance.instance_id
            instance_dict = {}
            if instance_type != scenenet_pb2.Instance.BACKGROUND:
                wnid = instance.semantic_wordnet_id
                wn_class = instance.semantic_english
                instance_dict['wordnet_id'] = wnid
                instance_dict['wordnet_class_name'] = wn_class
                if wnid in definitions.WNID_TO_COCO:
                    instance_dict['coco_id'] = definitions.WNID_TO_COCO[wnid]
                else:
                    continue  # only save instances that are positive instances and not background
            
                instances_traj_dict[instance_id] = instance_dict
        
        # iterate through images/frames
        for view in trajectory.views:
            frame_num = view.frame_num
            instance_file = f"{path}/instance/{frame_num}.png"
            file_names_photos_traj.append(f"{path}/photo/{frame_num}.jpg")
            file_names_instances_traj.append(instance_file)
            instances_view = []
            
            # load instance file
            instance_image = scipy.misc.imread(instance_file)
            for instance_id in instances_traj_dict:
                instance_local = np.copy(instance_image)
                instance_local[instance_local != instance_id] = 0
                instance_local[instance_local == instance_id] = 1
                coordinates = ndimage.find_objects(instance_local)
                if coordinates is None or not coordinates:  # the current instance was not in this frame
                    continue
                else:
                    coordinates = coordinates[0]  # extract the coords of the one object
                
                x = coordinates[1]
                y = coordinates[0]
                xmin, xmax = x.start, x.stop
                ymin, ymax = y.start, y.stop
                instance = instances_traj_dict[instance_id].copy()
                instance['bbox'] = (xmin, ymin, xmax, ymax)
                instances_view.append(instance)
            
            instances_traj.append(instances_view)
        
        file_names_photos.append(file_names_photos_traj)
        file_names_instances.append(file_names_instances_traj)
        instances.append(instances_traj)
    
    return file_names_photos, file_names_instances, instances
