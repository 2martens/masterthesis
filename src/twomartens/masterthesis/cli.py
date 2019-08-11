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
Provides CLI actions.

Functions:
    config(...): handles the config component
    train(...): trains a network
    test(...): evaluates a network
    val(...): runs predictions on the validation data
    prepare(...): prepares the SceneNet ground truth data
"""
import argparse
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generator
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import math
import numpy as np
import tensorflow as tf

from twomartens.masterthesis import config as conf


def config(args: argparse.Namespace) -> None:
    _config_execute_action(args, conf.get_property, conf.set_property, conf.list_property_values)


def prepare(args: argparse.Namespace) -> None:
    import pickle
    
    from twomartens.masterthesis import data
    
    file_names_photos, file_names_instances, \
        instances = data.prepare_scenenet_data(conf.get_property("Paths.scenenet"),
                                               args.protobuf_path)
    with open(f"{conf.get_property('Paths.scenenet_gt')}/"
              f"{args.ground_truth_path}/photo_paths.bin", "wb") as file:
        pickle.dump(file_names_photos, file)
    
    with open(f"{conf.get_property('Paths.scenenet_gt')}/"
              f"{args.ground_truth_path}/instance_paths.bin", "wb") as file:
        pickle.dump(file_names_instances, file)
    
    with open(f"{conf.get_property('Paths.scenenet_gt')}/"
              f"{args.ground_truth_path}/instances.bin", "wb") as file:
        pickle.dump(instances, file)


def train(args: argparse.Namespace) -> None:
    _train_execute_action(args, _ssd_train, _auto_encoder_train)


def test(args: argparse.Namespace) -> None:
    if args.network == "ssd" or args.network == "bayesian_ssd":
        _ssd_test(args)
    elif args.network == "auto_encoder":
        _auto_encoder_test(args)


def evaluate(args: argparse.Namespace) -> None:
    if args.network == "ssd":
        _ssd_evaluate(args)
    else:
        raise NotImplementedError


def visualise(args: argparse.Namespace) -> None:
    from twomartens.masterthesis.ssd_keras.eval_utils import coco_utils
    
    output_path, coco_path, ground_truth_path = _visualise_get_config_values(conf.get_property)
    output_path, annotation_file_train, \
        ground_truth_path = _visualise_prepare_paths(args, output_path, coco_path,
                                                     ground_truth_path)
    file_names, instances, \
        cats_to_classes, cats_to_names = _visualise_load_gt(ground_truth_path, annotation_file_train,
                                                            coco_utils.get_coco_category_maps)
    
    _visualise_gt(args, file_names, instances, cats_to_classes, cats_to_names, output_path)


def visualise_metrics(args: argparse.Namespace) -> None:
    output_path, evaluation_path = _visualise_metrics_get_config_values(conf.get_property)
    output_path, metrics_file = _visualise_metrics_prepare_paths(args, output_path, evaluation_path)
    _visualise_metrics(_visualise_precision_recall, _visualise_ose_f1,
                       output_path, metrics_file)


def measure_mapping(args: argparse.Namespace) -> None:
    from twomartens.masterthesis.ssd_keras.eval_utils import coco_utils
    
    output_path, coco_path, ground_truth_path = _measure_get_config_values(conf.get_property)
    output_path, annotation_file_train, ground_truth_path = _measure_prepare_paths(args, output_path, coco_path,
                                                                                   ground_truth_path)
    instances, cats_to_classes, cats_to_names = _measure_load_gt(ground_truth_path, annotation_file_train,
                                                                 coco_utils.get_coco_category_maps)
    nr_digits = _get_nr_digits(instances)
    _measure(instances, cats_to_classes, cats_to_names, nr_digits, output_path)


def _measure(instances: Sequence[Sequence[Sequence[dict]]],
             cats_to_classes: Dict[int, int],
             cats_to_names: Dict[int, str],
             nr_digits: int, output_path: str) -> None:
    import pickle
    
    with open(f"{output_path}/names.bin", "wb") as file:
        pickle.dump(cats_to_names, file)
    
    for i, trajectory in enumerate(instances):
        counts = {cat_id: 0 for cat_id in cats_to_classes.keys()}
        usable_frames = 0
        for labels in trajectory:
            if labels:
                usable_frames += 1
            for instance in labels:
                counts[instance['coco_id']] += 1
        
        pickle_content = {"usable_frames": usable_frames, "instance_counts": counts}
        with open(f"{output_path}/{str(i).zfill(nr_digits)}.bin", "wb") as file:
            pickle.dump(pickle_content, file)


def _config_execute_action(args: argparse.Namespace, on_get: callable,
                           on_set: callable, on_list: callable) -> None:
    if args.action == "get":
        print(str(on_get(args.property)))
    elif args.action == "set":
        on_set(args.property, args.value)
    elif args.action == "list":
        on_list()


def _train_execute_action(args: argparse.Namespace, on_ssd: callable, on_auto_encoder: callable) -> None:
    if args.network == "ssd" or args.network == "bayesian_ssd":
        on_ssd(args)
    elif args.network == "auto_encoder":
        on_auto_encoder(args)


def _ssd_train(args: argparse.Namespace) -> None:
    from twomartens.masterthesis import data
    from twomartens.masterthesis import debug
    from twomartens.masterthesis import ssd
    
    from twomartens.masterthesis.ssd_keras.eval_utils import coco_utils
    from twomartens.masterthesis.ssd_keras.models import keras_ssd300
    from twomartens.masterthesis.ssd_keras.models import keras_ssd300_dropout
    
    _init_eager_mode()
    
    batch_size, image_size, learning_rate, steps_per_val_epoch, nr_classes, \
        dropout_rate, top_k, nr_trajectories, \
        coco_path, summary_path, weights_path, train_gt_path, val_gt_path, \
        save_train_images, save_summaries = _ssd_train_get_config_values(conf.get_property)
    
    use_dropout = _ssd_is_dropout(args)
    summary_path, weights_path, \
        pre_trained_weights_file = _ssd_train_prepare_paths(args, summary_path, weights_path)
    
    file_names_train, instances_train, \
        file_names_val, instances_val = _ssd_train_load_gt(train_gt_path, val_gt_path)
    
    ssd_model, predictor_sizes = ssd.get_model(use_dropout,
                                               keras_ssd300_dropout.ssd_300_dropout,
                                               keras_ssd300.ssd_300,
                                               image_size,
                                               nr_classes,
                                               "training",
                                               dropout_rate,
                                               top_k,
                                               pre_trained_weights_file)
    loss_func = ssd.get_loss_func()
    ssd.compile_model(ssd_model, learning_rate, loss_func)
    
    train_generator, train_length, train_debug_generator, \
        val_generator, val_length, val_debug_generator = _ssd_train_get_generators(args,
                                                                                   data.load_scenenet_data,
                                                                                   file_names_train,
                                                                                   instances_train,
                                                                                   file_names_val,
                                                                                   instances_val,
                                                                                   coco_path,
                                                                                   batch_size,
                                                                                   image_size,
                                                                                   nr_trajectories,
                                                                                   predictor_sizes)
    
    _ssd_debug_save_images(args, save_train_images,
                           debug.save_ssd_train_images, coco_utils.get_coco_category_maps,
                           summary_path, coco_path,
                           image_size, train_debug_generator)
    
    nr_batches_train = _get_nr_batches(train_length, batch_size)
    tensorboard_callback = _ssd_get_tensorboard_callback(args, save_summaries, summary_path)
    
    history = _ssd_train_call(
        args,
        ssd.train,
        train_generator,
        nr_batches_train,
        val_generator,
        steps_per_val_epoch,
        ssd_model,
        weights_path,
        tensorboard_callback
    )
    
    _ssd_save_history(summary_path, history)


def _ssd_test(args: argparse.Namespace) -> None:
    from twomartens.masterthesis import data
    from twomartens.masterthesis import ssd

    from twomartens.masterthesis.ssd_keras.models import keras_ssd300
    from twomartens.masterthesis.ssd_keras.models import keras_ssd300_dropout
    
    _init_eager_mode()
    
    batch_size, image_size, learning_rate, \
        forward_passes_per_image, nr_classes, confidence_threshold, iou_threshold, \
        dropout_rate, \
        use_entropy_threshold, entropy_threshold_min, entropy_threshold_max, \
        use_coco, \
        top_k, nr_trajectories, test_pretrained, \
        coco_path, output_path, weights_path, ground_truth_path = _ssd_test_get_config_values(args, conf.get_property)
    
    use_dropout = _ssd_is_dropout(args)
    
    output_path, weights_file = _ssd_test_prepare_paths(args, output_path,
                                                        weights_path, test_pretrained)
    
    file_names, instances = _ssd_test_load_gt(ground_truth_path)

    ssd_model, predictor_sizes = ssd.get_model(use_dropout,
                                               keras_ssd300_dropout.ssd_300_dropout,
                                               keras_ssd300.ssd_300,
                                               image_size,
                                               nr_classes,
                                               "training",
                                               dropout_rate,
                                               top_k,
                                               weights_file)

    loss_func = ssd.get_loss_func()
    ssd.compile_model(ssd_model, learning_rate, loss_func)

    test_generator, length_dataset, test_debug_generator = _ssd_test_get_generators(args,
                                                                                    use_coco,
                                                                                    data.load_coco_val_ssd,
                                                                                    data.load_scenenet_data,
                                                                                    file_names,
                                                                                    instances,
                                                                                    coco_path,
                                                                                    batch_size,
                                                                                    image_size,
                                                                                    nr_trajectories,
                                                                                    predictor_sizes)
    
    nr_digits = _get_nr_digits(length_dataset, batch_size)
    steps_per_epoch = _get_nr_batches(length_dataset, batch_size)
    ssd.predict(test_generator,
                ssd_model,
                steps_per_epoch,
                image_size,
                batch_size,
                forward_passes_per_image,
                use_entropy_threshold,
                entropy_threshold_min,
                entropy_threshold_max,
                confidence_threshold,
                iou_threshold,
                output_path,
                coco_path,
                use_dropout,
                nr_digits)


def _ssd_evaluate(args: argparse.Namespace) -> None:
    from twomartens.masterthesis import debug
    from twomartens.masterthesis import evaluate

    from twomartens.masterthesis.ssd_keras.bounding_box_utils import bounding_box_utils
    from twomartens.masterthesis.ssd_keras.eval_utils import coco_utils
    
    _init_eager_mode()
    
    batch_size, image_size, iou_threshold, nr_classes, \
        evaluation_path, output_path, coco_path = _ssd_evaluate_get_config_values(config_get=conf.get_property)
    
    output_path, evaluation_path, \
        result_file, label_file, filenames_file, \
        predictions_file, predictions_per_class_file, \
        predictions_glob_string, label_glob_string = _ssd_evaluate_prepare_paths(args,
                                                                                 output_path,
                                                                                 evaluation_path)
    
    labels, filenames = _ssd_evaluate_unbatch_dict(label_glob_string)
    _pickle(label_file, labels)
    _pickle(filenames_file, filenames)
    
    predictions = _ssd_evaluate_unbatch_list(predictions_glob_string)
    _pickle(predictions_file, predictions)

    _ssd_evaluate_save_images(filenames, predictions,
                              coco_utils.get_coco_category_maps, debug.save_ssd_train_images,
                              image_size, batch_size,
                              output_path, coco_path)
    
    predictions_per_class = evaluate.prepare_predictions(predictions, nr_classes)
    _pickle(predictions_per_class_file, predictions_per_class)

    number_gt_per_class = evaluate.get_number_gt_per_class(labels, nr_classes)

    true_positives, false_positives, \
        cum_true_positives, cum_false_positives, \
        open_set_error, cumulative_open_set_error, \
        cum_true_positives_overall, cum_false_positives_overall = evaluate.match_predictions(predictions_per_class,
                                                                                             labels,
                                                                                             bounding_box_utils.iou,
                                                                                             nr_classes, iou_threshold)
    
    cum_precisions, cum_recalls, \
        cum_precisions_micro, cum_recalls_micro, \
        cum_precisions_macro, cum_recalls_macro = evaluate.get_precision_recall(number_gt_per_class,
                                                                                cum_true_positives,
                                                                                cum_false_positives,
                                                                                cum_true_positives_overall,
                                                                                cum_false_positives_overall,
                                                                                nr_classes)
    
    f1_scores, f1_scores_micro, f1_scores_macro = evaluate.get_f1_score(cum_precisions, cum_recalls,
                                                                        cum_precisions_micro, cum_recalls_micro,
                                                                        cum_precisions_macro, cum_recalls_macro,
                                                                        nr_classes)
    average_precisions = evaluate.get_mean_average_precisions(cum_precisions, cum_recalls, nr_classes)
    mean_average_precision = evaluate.get_mean_average_precision(average_precisions)
    
    results = _ssd_evaluate_get_results(true_positives,
                                        false_positives,
                                        cum_true_positives,
                                        cum_false_positives,
                                        cum_true_positives_overall,
                                        cum_false_positives_overall,
                                        cum_precisions,
                                        cum_recalls,
                                        cum_precisions_micro,
                                        cum_recalls_micro,
                                        cum_precisions_macro,
                                        cum_recalls_macro,
                                        f1_scores,
                                        f1_scores_micro,
                                        f1_scores_macro,
                                        average_precisions,
                                        mean_average_precision,
                                        open_set_error,
                                        cumulative_open_set_error)
    
    _pickle(result_file, results)


def _ssd_evaluate_save_images(filenames: Sequence[str], labels: Sequence[np.ndarray],
                              get_coco_cat_maps_func: callable, save_images: callable,
                              image_size: int, batch_size: int,
                              output_path: str, coco_path: str) -> None:
    
    save_images(filenames[:batch_size], labels[:batch_size], output_path, coco_path, image_size, get_coco_cat_maps_func)


def _visualise_gt(args: argparse.Namespace,
                  file_names: Sequence[Sequence[str]], instances: Sequence[Sequence[Sequence[dict]]],
                  cats_to_classes: Dict[int, int], cats_to_names: Dict[int, str],
                  output_path: str):
    from matplotlib import pyplot
    from PIL import Image
    
    colors = pyplot.cm.hsv(np.linspace(0, 1, 81)).tolist()
    
    i = 0
    nr_images = len(file_names[args.trajectory])
    nr_digits = math.ceil(math.log10(nr_images))
    for file_name, labels in zip(file_names[args.trajectory], instances[args.trajectory]):
        if not labels:
            continue
        
        # only loop through selected trajectory
        with Image.open(file_name) as image:
            figure = pyplot.figure()
            pyplot.imshow(image)
            
            current_axis = pyplot.gca()
            
            for instance in labels:
                bbox = instance['bbox']
                # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
                xmin = bbox[0]
                ymin = bbox[1]
                xmax = bbox[2]
                ymax = bbox[3]
                color = colors[cats_to_classes[int(instance['coco_id'])]]
                label = f"{cats_to_names[int(instance['coco_id'])]}: {instance['wordnet_class_name']}, " \
                    f"{instance['wordnet_id']}"
                current_axis.add_patch(
                    pyplot.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=color, fill=False, linewidth=2))
                current_axis.text(xmin, ymin, label, size='x-large', color='white',
                                  bbox={'facecolor': color, 'alpha': 1.0})
            pyplot.savefig(f"{output_path}/{str(i).zfill(nr_digits)}")
            pyplot.close(figure)
        
        i += 1


def _visualise_metrics(visualise_precision_recall: callable,
                       visualise_ose_f1: callable,
                       output_path: str,
                       metrics_file: str) -> None:
    import pickle
    
    with open(metrics_file, "rb") as file:
        metrics = pickle.load(file)
    
    precision_micro = metrics["cumulative_precision_micro"]
    recall_micro = metrics["cumulative_recall_micro"]
    visualise_precision_recall(precision_micro, recall_micro,
                               output_path, "micro")

    precision_macro = metrics["cumulative_precision_macro"]
    recall_macro = metrics["cumulative_recall_macro"]
    visualise_precision_recall(precision_macro, recall_macro,
                               output_path, "macro")

    f1_scores_micro = metrics["f1_scores_micro"]
    cumulative_ose = metrics["cumulative_open_set_error"]
    visualise_ose_f1(cumulative_ose, f1_scores_micro,
                     output_path, "micro")

    f1_scores_macro = metrics["f1_scores_macro"]
    visualise_ose_f1(cumulative_ose, f1_scores_macro,
                     output_path, "macro")

    # TODO add further metrics


def _init_eager_mode() -> None:
    tf.enable_eager_execution()
    

def _pickle(filename: str, content: Any) -> None:
    import pickle
    
    with open(filename, "wb") as file:
        pickle.dump(content, file)
        

def _get_nr_batches(data_length: int, batch_size: int) -> int:
    return int(math.floor(data_length / batch_size))


def _get_nr_digits(data_length: Union[int, Sequence], batch_size: int = 1) -> int:
    """
    
    Args:
        data_length: length of data or iterable with length
        batch_size: size of a batch if applicable

    Returns:
        number of digits required to print largest number
    """
    if type(data_length) is not int:
        data_length = len(data_length)
    return math.ceil(math.log10(math.ceil(data_length / batch_size)))
    

def _ssd_evaluate_unbatch_dict(glob_string: str) -> tuple:
    import glob
    import pickle
    
    unbatched_dict = None
    files = glob.glob(glob_string)
    files.sort()
    nr_keys = None
    for filename in files:
        with open(filename, "rb") as file:
            batched = pickle.load(file)
            if unbatched_dict is None:
                nr_keys = len(batched.keys())
                unbatched_dict = tuple([[] for _ in range(nr_keys)])
            
            batched = list(batched.values())
            
            for i in range(nr_keys):
                value = batched[i]
                unbatched_dict[i].extend(value)
    
    return unbatched_dict


def _ssd_evaluate_unbatch_list(glob_string: str) -> List[np.ndarray]:
    import glob
    import pickle
    
    unbatched = []
    files = glob.glob(glob_string)
    files.sort()
    for filename in files:
        with open(filename, "rb") as file:
            batched = pickle.load(file)
            unbatched.extend(batched)
    
    return unbatched


def _ssd_train_get_config_values(config_get: Callable[[str], Union[str, float, int, bool]]
                                ) -> Tuple[int, int, float, int, int, float, int, int,
                                           str, str, str, str, str,
                                           bool, bool]:
    
    batch_size = config_get("Parameters.batch_size")
    image_size = config_get("Parameters.ssd_image_size")
    learning_rate = config_get("Parameters.learning_rate")
    steps_per_val_epoch = config_get("Parameters.steps_per_val_epoch")
    nr_classes = config_get("Parameters.nr_classes")
    dropout_rate = config_get("Parameters.ssd_dropout_rate")
    top_k = config_get("Parameters.ssd_top_k")
    nr_trajectories = config_get("Parameters.nr_trajectories")
    
    coco_path = config_get("Paths.coco")
    summary_path = config_get("Paths.summaries")
    weights_path = config_get("Paths.weights")
    train_gt_path = config_get('Paths.scenenet_gt_train')
    val_gt_path = config_get('Paths.scenenet_gt_val')
    
    save_train_images = config_get("Debug.train_images")
    save_summaries = config_get("Debug.summaries")
    
    return (
        batch_size,
        image_size,
        learning_rate,
        steps_per_val_epoch,
        nr_classes,
        dropout_rate,
        top_k,
        nr_trajectories,
        #
        coco_path,
        summary_path,
        weights_path,
        train_gt_path,
        val_gt_path,
        #
        save_train_images,
        save_summaries
    )


def _ssd_test_get_config_values(args: argparse.Namespace,
                                config_get: Callable[[str], Union[str, float, int, bool]]
                                ) -> Tuple[int, int, float, int, int, float, float, float,
                                           bool, float, float,
                                           bool,
                                           int, int, bool,
                                           str, str, str, str]:
    
    batch_size = config_get("Parameters.batch_size")
    image_size = config_get("Parameters.ssd_image_size")
    learning_rate = config_get("Parameters.learning_rate")
    forward_passes_per_image = config_get("Parameters.ssd_forward_passes_per_image")
    nr_classes = config_get("Parameters.nr_classes")
    confidence_threshold = config_get("Parameters.ssd_confidence_threshold")
    iou_threshold = config_get("Parameters.ssd_iou_threshold")
    dropout_rate = config_get("Parameters.ssd_dropout_rate")
    use_entropy_threshold = config_get("Parameters.ssd_use_entropy_threshold")
    entropy_threshold_min = config_get("Parameters.ssd_entropy_threshold_min")
    entropy_threshold_max = config_get("Parameters.ssd_entropy_threshold_max")
    use_coco = config_get("Parameters.ssd_use_coco")
    top_k = config_get("Parameters.ssd_top_k")
    nr_trajectories = config_get("Parameters.nr_trajectories")
    test_pretrained = config_get("Parameters.ssd_test_pretrained")

    coco_path = config_get("Paths.coco")
    output_path = config_get("Paths.output")
    weights_path = config_get("Paths.weights")
    if args.debug:
        ground_truth_path = config_get("Paths.scenenet_gt_train")
    else:
        ground_truth_path = config_get("Paths.scenenet_gt_test")
    
    return (
        batch_size,
        image_size,
        learning_rate,
        forward_passes_per_image,
        nr_classes,
        confidence_threshold,
        iou_threshold,
        dropout_rate,
        #
        use_entropy_threshold,
        entropy_threshold_min,
        entropy_threshold_max,
        #
        use_coco,
        #
        top_k,
        nr_trajectories,
        test_pretrained,
        #
        coco_path,
        output_path,
        weights_path,
        ground_truth_path
    )


def _ssd_evaluate_get_config_values(config_get: Callable[[str], Union[str, int, float, bool]]
                                    ) -> Tuple[int, int, float, int,
                                               str, str, str]:
    batch_size = config_get("Parameters.batch_size")
    image_size = config_get("Parameters.ssd_image_size")
    iou_threshold = config_get("Parameters.ssd_iou_threshold")
    nr_classes = config_get("Parameters.nr_classes")
    
    evaluation_path = config_get("Paths.evaluation")
    output_path = config_get("Paths.output")
    coco_path = config_get("Paths.coco")
    
    return batch_size, image_size, iou_threshold, nr_classes, evaluation_path, output_path, coco_path


def _measure_get_config_values(config_get: Callable[[str], Union[str, int, float, bool]]
                               ) -> Tuple[str, str, str]:
    output_path = config_get("Paths.output")
    coco_path = config_get("Paths.coco")
    ground_truth_path = config_get("Paths.scenenet_gt")
    
    return output_path, coco_path, ground_truth_path


def _visualise_get_config_values(config_get: Callable[[str], Union[str, int, float, bool]]
                                 ) -> Tuple[str, str, str]:
    output_path = config_get("Paths.output")
    coco_path = config_get("Paths.coco")
    ground_truth_path = config_get("Paths.scenenet_gt")
    
    return output_path, coco_path, ground_truth_path


def _visualise_metrics_get_config_values(config_get: Callable[[str], Union[str, int, float, bool]]
                                         ) -> Tuple[str, str]:
    output_path = config_get("Paths.output")
    evaluation_path = config_get("Paths.evaluation")
    
    return output_path, evaluation_path


def _ssd_is_dropout(args: argparse.Namespace) -> bool:
    return False if args.network == "ssd" else True


def _ssd_train_prepare_paths(args: argparse.Namespace,
                             summary_path: str, weights_path: str) -> Tuple[str, str, str]:
    import os
    
    summary_path = f"{summary_path}/{args.network}/train/{args.iteration}"
    pre_trained_weights_file = f"{weights_path}/{args.network}/VGG_coco_SSD_300x300_iter_400000.h5"
    weights_path = f"{weights_path}/{args.network}/train/"
    
    os.makedirs(summary_path, exist_ok=True)
    os.makedirs(weights_path, exist_ok=True)
    
    return summary_path, weights_path, pre_trained_weights_file


def _ssd_test_prepare_paths(args: argparse.Namespace,
                            output_path: str, weights_path: str,
                            test_pretrained: bool) -> Tuple[str, str]:
    import os
    
    output_path = f"{output_path}/{args.network}/test/{args.iteration}/"
    checkpoint_path = f"{weights_path}/{args.network}/train/{args.train_iteration}"
    if test_pretrained:
        weights_file = f"{weights_path}/ssd/VGG_coco_SSD_300x300_iter_400000_subsampled.h5"
    else:
        weights_file = f"{checkpoint_path}/ssd300_weights.h5"
    
    os.makedirs(output_path, exist_ok=True)
    
    return output_path, weights_file


def _ssd_evaluate_prepare_paths(args: argparse.Namespace,
                                output_path: str, evaluation_path: str) -> Tuple[str, str, str,
                                                                                 str, str, str, str,
                                                                                 str, str]:
    import os
    
    output_path = f"{output_path}/{args.network}/test/{args.iteration}"
    evaluation_path = f"{evaluation_path}/{args.network}"
    result_file = f"{evaluation_path}/results-{args.iteration}.bin"
    label_file = f"{output_path}/labels.bin"
    filenames_file = f"{output_path}/filenames.bin"
    predictions_file = f"{output_path}/predictions.bin"
    predictions_per_class_file = f"{output_path}/predictions_class.bin"
    prediction_glob_string = f"{output_path}/*ssd_prediction*"
    label_glob_string = f"{output_path}/*ssd_label*"
    
    os.makedirs(evaluation_path, exist_ok=True)
    
    return (
        output_path, evaluation_path,
        result_file, label_file, filenames_file, predictions_file, predictions_per_class_file,
        prediction_glob_string, label_glob_string
    )


def _measure_prepare_paths(args: argparse.Namespace,
                           output_path: str, coco_path: str,
                           ground_truth_path: str) -> Tuple[str, str, str]:
    import os

    annotation_file_train = f"{coco_path}/annotations/instances_train2014.json"
    output_path = f"{output_path}/measure/{args.tarball_id}"
    ground_truth_path = f"{ground_truth_path}/{args.tarball_id}"

    os.makedirs(output_path, exist_ok=True)
    
    return output_path, annotation_file_train, ground_truth_path


def _visualise_prepare_paths(args: argparse.Namespace,
                             output_path: str, coco_path: str,
                             gt_path: str) -> Tuple[str, str, str]:
    import os
    
    output_path = f"{output_path}/visualise/{args.trajectory}"
    annotation_file_train = f"{coco_path}/annotations/instances_train2014.json"
    ground_truth_path = f"{gt_path}/{args.tarball_id}/"
    
    os.makedirs(output_path, exist_ok=True)

    return output_path, annotation_file_train, ground_truth_path


def _visualise_metrics_prepare_paths(args: argparse.Namespace,
                                     output_path: str,
                                     evaluation_path: str) -> Tuple[str, str]:
    import os
    
    metrics_file = f"{evaluation_path}/{args.network}/results-{args.iteration}.bin"
    output_path = f"{output_path}/{args.network}/visualise/{args.iteration}"
    
    os.makedirs(output_path, exist_ok=True)
    
    return output_path, metrics_file


def _ssd_train_load_gt(train_gt_path: str, val_gt_path: str
                       ) -> Tuple[Sequence[Sequence[str]],
                                  Sequence[Sequence[Sequence[dict]]],
                                  Sequence[Sequence[str]],
                                  Sequence[Sequence[Sequence[dict]]]]:

    import pickle

    with open(f"{train_gt_path}/photo_paths.bin", "rb") as file:
        file_names_train = pickle.load(file)
    with open(f"{train_gt_path}/instances.bin", "rb") as file:
        instances_train = pickle.load(file)
    with open(f"{val_gt_path}/photo_paths.bin", "rb") as file:
        file_names_val = pickle.load(file)
    with open(f"{val_gt_path}/instances.bin", "rb") as file:
        instances_val = pickle.load(file)
        
    return file_names_train, instances_train, file_names_val, instances_val


def _ssd_test_load_gt(gt_path: str) -> Tuple[Sequence[Sequence[str]],
                                             Sequence[Sequence[Sequence[dict]]]]:
    import pickle
    
    with open(f"{gt_path}/photo_paths.bin", "rb") as file:
        file_names = pickle.load(file)
    with open(f"{gt_path}/instances.bin", "rb") as file:
        instances = pickle.load(file)
        
    return file_names, instances


def _measure_load_gt(gt_path: str, annotation_file_train: str,
                     get_coco_cat_maps_func: callable) -> Tuple[Sequence[Sequence[Sequence[dict]]],
                                                                Dict[int, int],
                                                                Dict[int, str]]:
    import pickle
    
    with open(f"{gt_path}/instances.bin", "rb") as file:
        instances = pickle.load(file)
    cats_to_classes, _, cats_to_names, _ = get_coco_cat_maps_func(annotation_file_train)
    
    return instances, cats_to_classes, cats_to_names


def _visualise_load_gt(gt_path: str, annotation_file_train: str,
                       get_coco_cat_maps_func: callable) -> Tuple[Sequence[Sequence[str]],
                                                                  Sequence[Sequence[Sequence[dict]]],
                                                                  Dict[int, int],
                                                                  Dict[int, str]]:
    
    import pickle
    with open(f"{gt_path}/photo_paths.bin", "rb") as file:
        file_names = pickle.load(file)
    with open(f"{gt_path}/instances.bin", "rb") as file:
        instances = pickle.load(file)

    cats_to_classes, _, cats_to_names, _ = get_coco_cat_maps_func(annotation_file_train)
    
    return file_names, instances, cats_to_classes, cats_to_names


def _ssd_train_get_generators(args: argparse.Namespace,
                              load_data: callable,
                              file_names_train: Sequence[Sequence[str]],
                              instances_train: Sequence[Sequence[Sequence[dict]]],
                              file_names_val: Sequence[Sequence[str]],
                              instances_val: Sequence[Sequence[Sequence[dict]]],
                              coco_path: str,
                              batch_size: int,
                              image_size: int,
                              nr_trajectories: int,
                              predictor_sizes: Sequence[Sequence[int]]) -> Tuple[Generator, int, Generator, Generator, int, Generator]:
    
    if nr_trajectories == -1:
        nr_trajectories = None
        
    train_generator, train_length, train_debug_generator = \
        load_data(file_names_train, instances_train, coco_path,
                  predictor_sizes=predictor_sizes,
                  batch_size=batch_size,
                  image_size=image_size,
                  training=True, evaluation=False, augment=False,
                  debug=args.debug,
                  nr_trajectories=nr_trajectories)
    
    val_generator, val_length, val_debug_generator = \
        load_data(file_names_val, instances_val, coco_path,
                  predictor_sizes=predictor_sizes,
                  batch_size=batch_size,
                  image_size=image_size,
                  training=False, evaluation=False, augment=False,
                  debug=args.debug,
                  nr_trajectories=nr_trajectories)
    
    return (
        train_generator, train_length, train_debug_generator,
        val_generator, val_length, val_debug_generator
    )


def _ssd_test_get_generators(args: argparse.Namespace,
                             use_coco: bool,
                             load_data_coco: callable,
                             load_data_scenenet: callable,
                             file_names: Sequence[Sequence[str]],
                             instances: Sequence[Sequence[Sequence[dict]]],
                             coco_path: str,
                             batch_size: int,
                             image_size: int,
                             nr_trajectories: int,
                             predictor_sizes: Sequence[Sequence[int]]) -> Tuple[Generator, int, Generator]:
    
    from twomartens.masterthesis import data
    
    if nr_trajectories == -1:
        nr_trajectories = None
    
    if use_coco:
        generator, length, debug_generator = load_data_coco(data.clean_dataset,
                                                            data.group_bboxes_to_images,
                                                            coco_path,
                                                            batch_size,
                                                            image_size,
                                                            training=False, evaluation=True, augment=False,
                                                            debug=args.debug,
                                                            predictor_sizes=predictor_sizes)
    else:
        generator, length, debug_generator = load_data_scenenet(file_names, instances, coco_path,
                                                                predictor_sizes=predictor_sizes,
                                                                batch_size=batch_size,
                                                                image_size=image_size,
                                                                training=False, evaluation=True, augment=False,
                                                                debug=args.debug,
                                                                nr_trajectories=nr_trajectories)
    
    return generator, length, debug_generator


def _ssd_debug_save_images(args: argparse.Namespace, save_images_on_debug: bool,
                           save_images: callable, get_coco_cat_maps_func: callable,
                           summary_path: str, coco_path: str,
                           image_size: int,
                           train_generator: Generator) -> None:
    
    if args.debug and save_images_on_debug:
        train_data = next(train_generator)
        train_images = train_data[0]
        train_labels = train_data[1]
        train_labels_not_encoded = train_data[2]
        
        save_images(train_images, train_labels_not_encoded,
                    summary_path, coco_path, image_size,
                    get_coco_cat_maps_func, "before-encoding")

        save_images(train_images, train_labels,
                    summary_path, coco_path, image_size,
                    get_coco_cat_maps_func, "after-encoding")


def _ssd_get_tensorboard_callback(args: argparse.Namespace, save_summaries_on_debug: bool,
                                  summary_path: str) -> Union[None, tf.keras.callbacks.TensorBoard]:
    
    if args.debug and save_summaries_on_debug:
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=summary_path
        )
    else:
        tensorboard_callback = None
    
    return tensorboard_callback


def _ssd_train_call(args: argparse.Namespace, train_function: callable,
                    train_generator: Generator, nr_batches_train: int,
                    val_generator: Generator, nr_batches_val: int,
                    model: tf.keras.models.Model,
                    weights_path: str,
                    tensorboard_callback: Optional[tf.keras.callbacks.TensorBoard]) -> tf.keras.callbacks.History:
    
    history = train_function(
        train_generator,
        nr_batches_train,
        val_generator,
        nr_batches_val,
        model,
        weights_path,
        args.iteration,
        initial_epoch=0,
        nr_epochs=args.num_epochs,
        tensorboard_callback=tensorboard_callback
    )
    
    return history


def _ssd_save_history(summary_path: str, history: tf.keras.callbacks.History) -> None:
    import pickle
    
    with open(f"{summary_path}/history", "wb") as file:
        pickle.dump(history.history, file)


def _ssd_evaluate_get_results(true_positives: Sequence[np.ndarray],
                              false_positives: Sequence[np.ndarray],
                              cum_true_positives: Sequence[np.ndarray],
                              cum_false_positives: Sequence[np.ndarray],
                              cum_true_positives_micro: np.ndarray,
                              cum_false_positives_micro: np.ndarray,
                              cum_precisions: Sequence[np.ndarray],
                              cum_recalls: Sequence[np.ndarray],
                              cum_precision_micro: np.ndarray,
                              cum_recall_micro: np.ndarray,
                              cum_precision_macro: np.ndarray,
                              cum_recall_macro: np.ndarray,
                              f1_scores: Sequence[np.ndarray],
                              f1_scores_micro: np.ndarray,
                              f1_scores_macro: np.ndarray,
                              average_precisions: Sequence[float],
                              mean_average_precision: float,
                              open_set_error: np.ndarray,
                              cumulative_open_set_error: np.ndarray
                              ) -> Dict[str, Union[np.ndarray, float, int]]:
    results = {
        "true_positives":             true_positives,
        "false_positives":            false_positives,
        "cumulative_true_positives":  cum_true_positives,
        "cumulative_false_positives": cum_false_positives,
        "cumulative_true_positives_micro": cum_true_positives_micro,
        "cumulative_false_positives_micro": cum_false_positives_micro,
        "cumulative_precisions":      cum_precisions,
        "cumulative_recalls":         cum_recalls,
        "cumulative_precision_micro": cum_precision_micro,
        "cumulative_recall_micro":    cum_recall_micro,
        "cumulative_precision_macro": cum_precision_macro,
        "cumulative_recall_macro":    cum_recall_macro,
        "f1_scores":                  f1_scores,
        "f1_scores_micro":            f1_scores_micro,
        "f1_scores_macro":            f1_scores_macro,
        "mean_average_precisions":    average_precisions,
        "mean_average_precision":     mean_average_precision,
        "open_set_error":             open_set_error,
        "cumulative_open_set_error":  cumulative_open_set_error
    }
    
    return results


def _visualise_precision_recall(precision: np.ndarray, recall: np.ndarray,
                                output_path: str, file_suffix: str) -> None:
    from matplotlib import pyplot
    
    figure = pyplot.figure()
    
    pyplot.ylabel("precision")
    pyplot.xlabel("recall")
    pyplot.plot(recall, precision)
    
    pyplot.savefig(f"{output_path}/precision-recall-{file_suffix}.png")
    pyplot.close(figure)


def _visualise_ose_f1(open_set_error: np.ndarray, f1_scores: np.ndarray,
                      output_path: str, file_suffix: str) -> None:
    from matplotlib import pyplot
    
    figure = pyplot.figure()

    pyplot.ylabel("absolute ose")
    pyplot.xlabel("f1 score")
    pyplot.plot(f1_scores, open_set_error)
    
    pyplot.savefig(f"{output_path}/ose-f1-{file_suffix}.png")
    pyplot.close(figure)


def _auto_encoder_train(args: argparse.Namespace) -> None:
    import os
    
    import tensorflow as tf
    from tensorflow.python.ops import summary_ops_v2
    
    from twomartens.masterthesis import data
    from twomartens.masterthesis.aae import train
    
    tf.enable_eager_execution()
    coco_path = args.coco_path
    category = args.category
    batch_size = 16
    image_size = 256
    coco_data = data.load_coco_train(coco_path, category, num_epochs=args.num_epochs, batch_size=batch_size,
                                     resized_shape=(image_size, image_size))
    summary_path = conf.get_property("Paths.summary")
    summary_path = f"{summary_path}/{args.network}/train/category-{category}/{args.iteration}"
    train_summary_writer = summary_ops_v2.create_file_writer(
        summary_path
    )
    os.makedirs(summary_path, exist_ok=True)
    
    weights_path = conf.get_property("Paths.weights")
    weights_path = f"{weights_path}/{args.network}/category-{category}"
    os.makedirs(weights_path, exist_ok=True)
    if args.debug:
        with train_summary_writer.as_default():
            train.train_simple(coco_data, iteration=args.iteration,
                               weights_prefix=weights_path,
                               zsize=16, lr=0.0001, verbose=args.verbose, image_size=image_size,
                               channels=3, train_epoch=args.num_epochs, batch_size=batch_size)
    else:
        train.train_simple(coco_data, iteration=args.iteration,
                           weights_prefix=weights_path,
                           zsize=16, lr=0.0001, verbose=args.verbose, image_size=image_size,
                           channels=3, train_epoch=args.num_epochs, batch_size=batch_size)


def _auto_encoder_test(args: argparse.Namespace) -> None:
    import os
    
    import tensorflow as tf
    from tensorflow.python.ops import summary_ops_v2
    
    from twomartens.masterthesis import data
    from twomartens.masterthesis.aae import run
    
    tf.enable_eager_execution()
    coco_path = conf.get_property("Paths.coco")
    category = args.category
    category_trained = args.category_trained
    batch_size = 16
    image_size = 256
    coco_data = data.load_coco_val(coco_path, category, num_epochs=1,
                                   batch_size=batch_size, resized_shape=(image_size, image_size))
    
    summary_path = conf.get_property("Paths.summary")
    summary_path = f"{summary_path}/{args.network}/val/category-{category}/{args.iteration}"
    os.makedirs(summary_path, exist_ok=True)
    use_summary_writer = summary_ops_v2.create_file_writer(
        summary_path
    )
    
    weights_path = conf.get_property("Paths.weights")
    weights_path = f"{weights_path}/{args.network}/category-{category_trained}"
    os.makedirs(weights_path, exist_ok=True)
    if args.debug:
        with use_summary_writer.as_default():
            run.run_simple(coco_data, iteration=args.iteration_trained,
                           weights_prefix=weights_path,
                           zsize=16, verbose=args.verbose, channels=3, batch_size=batch_size,
                           image_size=image_size)
    else:
        run.run_simple(coco_data, iteration=args.iteration_trained,
                       weights_prefix=weights_path,
                       zsize=16, verbose=args.verbose, channels=3, batch_size=batch_size,
                       image_size=image_size)
