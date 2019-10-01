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
    prepare(...): prepares the SceneNet ground truth data
    train(...): trains a network
    test(...): tests a network
    evaluate(...): evaluates prediction results
    visualise(...): visualises ground truth
    visualise_metrics(...): visualises evaluation results
    visualise_all(...): creates figure for specific variants and thresholds
    measure_mapping(...): measures the number of instances per COCO class
"""
import argparse
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generator
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

import math
import numpy as np
import tensorflow as tf
from attributedict.collections import AttributeDict

from twomartens.masterthesis import config as conf


def config(args: argparse.Namespace) -> None:
    """Executes the config action from the CLI."""
    _config_execute_action(args, conf.get_property, conf.set_property, conf.list_property_values)


def prepare(args: argparse.Namespace) -> None:
    """Executes the prepare action from the CLI."""
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
    """Executes the train action from the CLI."""
    _train_execute_action(args, _ssd_train)


def test(args: argparse.Namespace) -> None:
    """Executes the test action from the CLI."""
    if args.network == "ssd" or args.network == "bayesian_ssd":
        _ssd_test(args)


def evaluate(args: argparse.Namespace) -> None:
    """Executes the evaluate action from the CLI."""
    if args.network == "ssd" or args.network == "bayesian_ssd":
        _ssd_evaluate(args)
    else:
        raise NotImplementedError


def visualise(args: argparse.Namespace) -> None:
    """Executes the visualise action from the CLI."""
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
    """Executes the visualise_metrics action from the CLI."""
    conf_obj = conf.Config()
    output_path, metrics_file = _visualise_metrics_prepare_paths(args, conf_obj)
    _visualise_metrics(_visualise_precision_recall, _visualise_ose_f1,
                       conf_obj,
                       output_path, metrics_file)


def visualise_all(args: argparse.Namespace) -> None:
    """Executes the visualise_all action from the CLI."""
    conf_obj = conf.Config()
    output_path, metrics_files = _visualise_all_prepare_paths(args, conf_obj)
    _visualise_all(_visualise_precision_recall_all, _visualise_ose_f1_all,
                   output_path, metrics_files)
    

def measure_mapping(args: argparse.Namespace) -> None:
    """Executes the measure_mapping action from the CLI."""
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


def _train_execute_action(args: argparse.Namespace, on_ssd: callable) -> None:
    if args.network == "ssd" or args.network == "bayesian_ssd":
        on_ssd(args)


def _ssd_train(args: argparse.Namespace) -> None:
    from twomartens.masterthesis import data
    from twomartens.masterthesis import plotting
    from twomartens.masterthesis import ssd
    
    from twomartens.masterthesis.ssd_keras.eval_utils import coco_utils
    from twomartens.masterthesis.ssd_keras.models import keras_ssd300
    from twomartens.masterthesis.ssd_keras.models import keras_ssd300_dropout
    
    _init_eager_mode()
    
    conf_obj = conf.Config()
    
    use_bayesian = _ssd_is_bayesian(args)
    paths = _ssd_train_prepare_paths(args, conf_obj)
    
    ground_truth = _ssd_train_load_gt(conf_obj)
    
    ssd_model, predictor_sizes = ssd.get_model(use_bayesian=use_bayesian,
                                               bayesian_model=keras_ssd300_dropout.ssd_300_dropout,
                                               vanilla_model=keras_ssd300.ssd_300,
                                               conf_obj=conf_obj,
                                               mode="training",
                                               pre_trained_weights_file=paths.pre_trained_weights_file)
    loss_func = ssd.get_loss_func()
    ssd.compile_model(ssd_model, conf_obj.parameters.learning_rate, loss_func)
    
    generators = _ssd_train_get_generators(args, conf_obj,
                                           data.load_scenenet_data,
                                           ground_truth,
                                           predictor_sizes)
    
    _ssd_debug_save_images(args, conf_obj, paths,
                           plotting.save_ssd_train_images, coco_utils.get_coco_category_maps,
                           generators.train_debug_generator)
    
    nr_batches_train = _get_nr_batches(generators.train_length, conf_obj.parameters.batch_size)
    tensorboard_callback = _ssd_get_tensorboard_callback(args, conf_obj.debug.summaries, paths.summary_path)

    history = ssd.train(
        generators.train_generator,
        nr_batches_train,
        generators.val_generator,
        conf_obj.parameters.steps_per_val_epoch,
        ssd_model,
        paths.weights_path,
        args.iteration,
        initial_epoch=0,
        nr_epochs=args.num_epochs,
        tensorboard_callback=tensorboard_callback
    )
    
    _ssd_save_history(paths.summary_path, history)


def _ssd_test(args: argparse.Namespace) -> None:
    from twomartens.masterthesis import data
    from twomartens.masterthesis import ssd

    from twomartens.masterthesis.ssd_keras.models import keras_ssd300
    from twomartens.masterthesis.ssd_keras.models import keras_ssd300_dropout
    
    _init_eager_mode()
    
    conf_obj = conf.Config()
    use_bayesian = _ssd_is_bayesian(args)
    paths = _ssd_test_prepare_paths(args, conf_obj)
    ground_truth = _ssd_test_load_gt(conf_obj)

    ssd_model, predictor_sizes = ssd.get_model(use_bayesian=use_bayesian,
                                               bayesian_model=keras_ssd300_dropout.ssd_300_dropout,
                                               vanilla_model=keras_ssd300.ssd_300,
                                               conf_obj=conf_obj,
                                               mode="training",
                                               pre_trained_weights_file=paths.weights_file)

    loss_func = ssd.get_loss_func()
    ssd.compile_model(ssd_model, conf_obj.parameters.learning_rate, loss_func)

    generators = _ssd_test_get_generators(args,
                                          conf_obj,
                                          data.load_coco_val_ssd,
                                          data.load_scenenet_data,
                                          ground_truth,
                                          predictor_sizes)
    
    nr_digits = _get_nr_digits(generators.length, conf_obj.parameters.batch_size)
    steps_per_epoch = _get_nr_batches(generators.length, conf_obj.parameters.batch_size)
    ssd.predict(generator=generators.generator,
                model=ssd_model,
                conf_obj=conf_obj,
                steps_per_epoch=steps_per_epoch,
                use_bayesian=use_bayesian,
                nr_digits=nr_digits,
                output_path=paths.output_path)


def _ssd_evaluate(args: argparse.Namespace) -> None:
    _init_eager_mode()
    
    conf_obj = conf.Config()
    
    paths = _ssd_evaluate_prepare_paths(args, conf_obj)

    labels, filenames = _ssd_evaluate_unbatch_dict(paths.label_glob_string)
    _pickle(paths.label_file, labels)
    _pickle(paths.filenames_file, filenames)
    
    entropy_thresholds = _get_entropy_thresholds(conf_obj.parameters.ssd_entropy_threshold_min,
                                                 conf_obj.parameters.ssd_entropy_threshold_max)

    _ssd_evaluate_entropy_loop(conf_obj=conf_obj, paths=paths,
                               entropy_thresholds=entropy_thresholds,
                               labels=labels, filenames=filenames)


def _get_entropy_thresholds(min_threshold: float, max_threshold: float) -> List[float]:
    nr_steps = math.floor((max_threshold - min_threshold) * 10)
    entropy_thresholds = [round(i / 10 + min_threshold, 1) for i in range(nr_steps)]
    
    return entropy_thresholds


def _ssd_evaluate_entropy_loop(conf_obj: conf.Config, paths: AttributeDict,
                               entropy_thresholds: Sequence[float],
                               labels: Sequence[Sequence], filenames: Sequence[str]) -> None:
    
    from twomartens.masterthesis import plotting
    from twomartens.masterthesis import evaluate
    
    from twomartens.masterthesis.ssd_keras.bounding_box_utils import bounding_box_utils
    from twomartens.masterthesis.ssd_keras.eval_utils import coco_utils
    
    if not conf_obj.parameters.ssd_use_entropy_threshold:
        entropy_thresholds = [0]
        
    for entropy_threshold in entropy_thresholds:
    
        predictions = _ssd_evaluate_unbatch_list(f"{paths.predictions_glob_string}-{entropy_threshold}"
                                                 if conf_obj.parameters.ssd_use_entropy_threshold
                                                 else paths.predictions_glob_string)
        _pickle(f"{paths.predictions_file}-{entropy_threshold}.bin"
                if conf_obj.parameters.ssd_use_entropy_threshold else f"{paths.predictions_file}.bin", predictions)
        
        _ssd_evaluate_save_images(filenames, predictions,
                                  coco_utils.get_coco_category_maps, plotting.save_ssd_train_images,
                                  conf_obj, paths)
        
        predictions_per_class = evaluate.prepare_predictions(predictions, conf_obj.parameters.nr_classes)
        _pickle(f"{paths.predictions_per_class_file}-{entropy_threshold}.bin"
                if conf_obj.parameters.ssd_use_entropy_threshold
                else f"{paths.predictions_per_class_file}.bin", predictions_per_class)
        
        number_gt_per_class = evaluate.get_number_gt_per_class(labels, conf_obj.parameters.nr_classes)
        
        true_positives, false_positives, \
            cum_true_positives, cum_false_positives, \
            open_set_error, cumulative_open_set_error, \
            cum_true_positives_overall, \
            cum_false_positives_overall = evaluate.match_predictions(predictions_per_class,
                                                                     labels,
                                                                     bounding_box_utils.iou,
                                                                     conf_obj.parameters.nr_classes,
                                                                     conf_obj.parameters.ssd_iou_threshold)
        
        cum_precisions, cum_recalls, \
            cum_precisions_micro, cum_recalls_micro, \
            cum_precisions_macro, cum_recalls_macro = evaluate.get_precision_recall(number_gt_per_class,
                                                                                    cum_true_positives,
                                                                                    cum_false_positives,
                                                                                    cum_true_positives_overall,
                                                                                    cum_false_positives_overall,
                                                                                    conf_obj.parameters.nr_classes)
        
        f1_scores, f1_scores_micro, f1_scores_macro = evaluate.get_f1_score(cum_precisions, cum_recalls,
                                                                            cum_precisions_micro, cum_recalls_micro,
                                                                            cum_precisions_macro, cum_recalls_macro,
                                                                            conf_obj.parameters.nr_classes)
        average_precisions = evaluate.get_mean_average_precisions(cum_precisions, cum_recalls,
                                                                  conf_obj.parameters.nr_classes)
        mean_average_precision = evaluate.get_mean_average_precision(average_precisions)
        
        results = _ssd_evaluate_get_results(true_positives=true_positives,
                                            false_positives=false_positives,
                                            cumulative_true_positives=cum_true_positives,
                                            cumulative_false_positives=cum_false_positives,
                                            cumulative_true_positives_overall=cum_true_positives_overall,
                                            cumulative_false_positives_overall=cum_false_positives_overall,
                                            cumulative_precisions=cum_precisions,
                                            cumulative_recalls=cum_recalls,
                                            cumulative_precisions_micro=cum_precisions_micro,
                                            cumulative_recalls_micro=cum_recalls_micro,
                                            cumulative_precisions_macro=cum_precisions_macro,
                                            cumulative_recalls_macro=cum_recalls_macro,
                                            f1_scores=f1_scores,
                                            f1_scores_micro=f1_scores_micro,
                                            f1_scores_macro=f1_scores_macro,
                                            average_precisions=average_precisions,
                                            mean_average_precision=mean_average_precision,
                                            open_set_error=open_set_error,
                                            cumulative_open_set_error=cumulative_open_set_error)
        
        _pickle(f"{paths.result_file}-{entropy_threshold}.bin"
                if conf_obj.parameters.ssd_use_entropy_threshold else f"{paths.result_file}.bin", results)


def _ssd_evaluate_save_images(filenames: Sequence[str], labels: Sequence[np.ndarray],
                              get_coco_cat_maps_func: callable, save_images: callable,
                              conf_obj: conf.Config, paths: AttributeDict) -> None:
    
    save_images(filenames[:conf_obj.parameters.batch_size],
                labels[:conf_obj.parameters.batch_size],
                paths.output_path, conf_obj.paths.coco,
                conf_obj.parameters.ssd_image_size,
                get_coco_cat_maps_func)


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
                       conf_obj: conf.Config,
                       output_path: str,
                       metrics_file: str) -> None:
    import pickle
    
    use_entropy_threshold = conf_obj.parameters.ssd_use_entropy_threshold
    if use_entropy_threshold:
        entropy_thresholds = _get_entropy_thresholds(conf_obj.parameters.ssd_entropy_threshold_min,
                                                     conf_obj.parameters.ssd_entropy_threshold_max)
    else:
        entropy_thresholds = [0]
    
    for threshold in entropy_thresholds:
        with open(f"{metrics_file}-{threshold}.bin"
                  if use_entropy_threshold
                  else f"{metrics_file}.bin", "rb") as file:
            metrics = pickle.load(file)
        
        try:
            precision_micro = metrics["cumulative_precisions_micro"]
            recall_micro = metrics["cumulative_recalls_micro"]
        except KeyError:
            precision_micro = metrics["cumulative_precision_micro"]
            recall_micro = metrics["cumulative_recall_micro"]
        
        visualise_precision_recall(precision_micro, recall_micro,
                                   output_path, f"micro-{threshold}"
                                   if use_entropy_threshold else "micro")

        try:
            precision_macro = metrics["cumulative_precisions_macro"]
            recall_macro = metrics["cumulative_recalls_macro"]
        except KeyError:
            precision_macro = metrics["cumulative_precision_macro"]
            recall_macro = metrics["cumulative_recall_macro"]
        visualise_precision_recall(precision_macro, recall_macro,
                                   output_path, f"macro-{threshold}"
                                   if use_entropy_threshold else "macro")
    
        f1_scores_micro = metrics["f1_scores_micro"]
        cumulative_ose = metrics["cumulative_open_set_error"]
        visualise_ose_f1(cumulative_ose, f1_scores_micro,
                         output_path, f"micro-{threshold}"
                         if use_entropy_threshold else "micro")
    
        f1_scores_macro = metrics["f1_scores_macro"]
        visualise_ose_f1(cumulative_ose, f1_scores_macro,
                         output_path, f"macro-{threshold}"
                         if use_entropy_threshold else "macro")
        
        precision = metrics["cumulative_precisions"]
        recall = metrics["cumulative_recalls"]
        f1_scores = metrics["f1_scores"]
        
        class_scores = {}
        for i in range(1, conf_obj.parameters.nr_classes + 1):
            if not len(f1_scores[i]):
                continue
            
            max_f1_score_index = np.argmax(f1_scores[i])
            max_f1_score = f1_scores[i][max_f1_score_index]
            precision_at_max_f1 = precision[i][max_f1_score_index]
            recall_at_max_f1 = recall[i][max_f1_score_index]
            class_scores[i] = {
                "max_f1_score": max_f1_score,
                "precision_at_max_f1": precision_at_max_f1,
                "recall_at_max_f1": recall_at_max_f1
            }
        
        max_f1_score_micro_index = np.argmax(f1_scores_micro, axis=0)
        max_f1_score_micro = f1_scores_micro[max_f1_score_micro_index]
        precision_at_max_f1_micro = precision_micro[max_f1_score_micro_index]
        recall_at_max_f1_micro = recall_micro[max_f1_score_micro_index]
        ose_at_max_f1_micro = cumulative_ose[max_f1_score_micro_index]
        
        max_f1_score_macro_index = np.argmax(f1_scores_macro, axis=0)
        max_f1_score_macro = f1_scores_macro[max_f1_score_macro_index]
        precision_at_max_f1_macro = precision_macro[max_f1_score_macro_index]
        recall_at_max_f1_macro = recall_macro[max_f1_score_macro_index]
        ose_at_max_f1_macro = cumulative_ose[max_f1_score_macro_index]
        
        import json
        with open(f"{output_path}/scores-{threshold}.json"
                  if use_entropy_threshold else f"{output_path}/scores.json", "w") as file:
            
            json.dump({
                "max_f1_score_micro": max_f1_score_micro,
                "precision_at_max_f1_micro": precision_at_max_f1_micro,
                "recall_at_max_f1_micro": recall_at_max_f1_micro,
                "ose_at_max_f1_micro": int(ose_at_max_f1_micro),
                "max_f1_score_macro":        max_f1_score_macro,
                "precision_at_max_f1_macro": precision_at_max_f1_macro,
                "recall_at_max_f1_macro":    recall_at_max_f1_macro,
                "ose_at_max_f1_macro":       int(ose_at_max_f1_macro),
                "class_scores": class_scores
            }, file, indent=2)


def _visualise_all(visualise_precision_recall: callable,
                   visualise_ose_f1: callable,
                   output_path: str,
                   metrics_files: Sequence[str]) -> None:
    
    import pickle
    
    metrics_micro = []
    metrics_macro = []
    with open(f"{metrics_files[0]}.bin", "rb") as file:
        metrics_vanilla_ssd_001 = pickle.load(file)
        metrics_micro.append(metrics_vanilla_ssd_001)
        metrics_macro.append(metrics_vanilla_ssd_001)
    with open(f"{metrics_files[1]}.bin", "rb") as file:
        metrics_vanilla_ssd_02 = pickle.load(file)
        metrics_micro.append(metrics_vanilla_ssd_02)
        metrics_macro.append(metrics_vanilla_ssd_02)
    with open(f"{metrics_files[2]}-1.7.bin", "rb") as file:
        metrics_entropy_ssd_001_macro = pickle.load(file)
        metrics_macro.append(metrics_entropy_ssd_001_macro)
    with open(f"{metrics_files[2]}-2.4.bin", "rb") as file:
        metrics_entropy_ssd_001_micro = pickle.load(file)
        metrics_micro.append(metrics_entropy_ssd_001_micro)
    with open(f"{metrics_files[3]}-1.0.bin", "rb") as file:
        metrics_bayesian_ssd_no_do_no_nms_micro = pickle.load(file)
        metrics_micro.append(metrics_bayesian_ssd_no_do_no_nms_micro)
    with open(f"{metrics_files[3]}-1.5.bin", "rb") as file:
        metrics_bayesian_ssd_no_do_no_nms_macro = pickle.load(file)
        metrics_macro.append(metrics_bayesian_ssd_no_do_no_nms_macro)
    with open(f"{metrics_files[4]}-1.4.bin", "rb") as file:
        metrics_bayesian_ssd_no_do_micro = pickle.load(file)
        metrics_micro.append(metrics_bayesian_ssd_no_do_micro)
    with open(f"{metrics_files[4]}-1.5.bin", "rb") as file:
        metrics_bayesian_ssd_no_do_macro = pickle.load(file)
        metrics_macro.append(metrics_bayesian_ssd_no_do_macro)
    with open(f"{metrics_files[5]}-1.4.bin", "rb") as file:
        metrics_bayesian_ssd_do_09_micro = pickle.load(file)
        metrics_micro.append(metrics_bayesian_ssd_do_09_micro)
    with open(f"{metrics_files[5]}-1.7.bin", "rb") as file:
        metrics_bayesian_ssd_do_09_macro = pickle.load(file)
        metrics_macro.append(metrics_bayesian_ssd_do_09_macro)
    with open(f"{metrics_files[6]}-1.3.bin", "rb") as file:
        metrics_bayesian_ssd_do_05_micro = pickle.load(file)
        metrics_micro.append(metrics_bayesian_ssd_do_05_micro)
    with open(f"{metrics_files[6]}-2.0.bin", "rb") as file:
        metrics_bayesian_ssd_do_05_macro = pickle.load(file)
        metrics_macro.append(metrics_bayesian_ssd_do_05_macro)
    
    legend = [
        "Vanilla SSD, 0.01 conf thresh",
        "Vanilla SSD, 0.2 conf thresh",
        "Vanilla SSD, 0.01 conf thresh, entropy thresh",
        "Bay. SSD, 0.2 conf thresh, no dropout",
        "Bay. SSD, 0.2 conf thresh, no dropout, NMS",
        "Bay. SSD, 0.2 conf thresh, dropout keep rate 0.9",
        "Bay. SSD, 0.2 conf thresh, dropout keep rate 0.5",
    ]
    
    # micro
    precisions_micro = []
    recalls_micro = []
    f1_scores_micro = []
    ose_values_micro = []
    for metrics in metrics_micro:
        try:
            precisions_micro.append(metrics["cumulative_precisions_micro"])
            recalls_micro.append(metrics["cumulative_recalls_micro"])
        except KeyError:
            precisions_micro.append(metrics["cumulative_precision_micro"])
            recalls_micro.append(metrics["cumulative_recall_micro"])
        
        f1_scores_micro.append(metrics["f1_scores_micro"])
        ose_values_micro.append(metrics["cumulative_open_set_error"])
    
    visualise_precision_recall(precisions_micro,
                               recalls_micro,
                               legend,
                               output_path, "micro")

    visualise_ose_f1(ose_values_micro,
                     f1_scores_micro,
                     legend,
                     output_path, "micro")

    # macro
    precisions_macro = []
    recalls_macro = []
    f1_scores_macro = []
    ose_values_macro = []
    for metrics in metrics_macro:
        try:
            precisions_macro.append(metrics["cumulative_precisions_macro"])
            recalls_macro.append(metrics["cumulative_recalls_macro"])
        except KeyError:
            precisions_macro.append(metrics["cumulative_precision_macro"])
            recalls_macro.append(metrics["cumulative_recall_macro"])
        f1_scores_macro.append(metrics["f1_scores_macro"])
        ose_values_macro.append(metrics["cumulative_open_set_error"])

    visualise_precision_recall(precisions_macro,
                               recalls_macro,
                               legend,
                               output_path, "macro")

    visualise_ose_f1(ose_values_macro,
                     f1_scores_macro,
                     legend,
                     output_path, "macro")
    

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


def _ssd_evaluate_get_config_values(config_get: Callable[[str], Union[str, int, float, bool]]
                                    ) -> Tuple[int, int, float, int,
                                               bool, float, float,
                                               str, str, str]:
    batch_size = config_get("Parameters.batch_size")
    image_size = config_get("Parameters.ssd_image_size")
    iou_threshold = config_get("Parameters.ssd_iou_threshold")
    nr_classes = config_get("Parameters.nr_classes")
    
    use_entropy_threshold = config_get("Parameters.ssd_use_entropy_threshold")
    entropy_threshold_min = config_get("Parameters.ssd_entropy_threshold_min")
    entropy_threshold_max = config_get("Parameters.ssd_entropy_threshold_max")
    
    evaluation_path = config_get("Paths.evaluation")
    output_path = config_get("Paths.output")
    coco_path = config_get("Paths.coco")
    
    return (
        batch_size, image_size, iou_threshold, nr_classes,
        use_entropy_threshold, entropy_threshold_min, entropy_threshold_max,
        evaluation_path, output_path, coco_path
    )


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


def _ssd_is_bayesian(args: argparse.Namespace) -> bool:
    return False if args.network == "ssd" else True


def _ssd_train_prepare_paths(args: argparse.Namespace,
                             conf_obj: conf.Config) -> AttributeDict:
    import os
    
    summary_path = f"{conf_obj.paths.summary}/{args.network}/train/{args.iteration}"
    pre_trained_weights_file = f"{conf_obj.paths.weights}/{args.network}/VGG_coco_SSD_300x300_iter_400000.h5"
    weights_path = f"{conf_obj.paths.weights}/{args.network}/train/"
    
    os.makedirs(summary_path, exist_ok=True)
    os.makedirs(weights_path, exist_ok=True)
    
    return AttributeDict({
        "summary_path": summary_path,
        "weights_path": weights_path,
        "pre_trained_weights_file": pre_trained_weights_file
    })


def _ssd_test_prepare_paths(args: argparse.Namespace,
                            conf_obj: conf.Config) -> AttributeDict:
    import os
    
    output_path = f"{conf_obj.paths.output}/{args.network}/test/{args.iteration}/"
    checkpoint_path = f"{conf_obj.paths.weights}/{args.network}/train/{args.train_iteration}"
    if conf_obj.parameters.ssd_test_pretrained:
        weights_file = f"{conf_obj.paths.weights}/ssd/VGG_coco_SSD_300x300_iter_400000_subsampled.h5"
    else:
        weights_file = f"{checkpoint_path}/ssd300_weights.h5"
    
    os.makedirs(output_path, exist_ok=True)
    
    return AttributeDict({
        "output_path": output_path,
        "weights_file": weights_file
    })


def _ssd_evaluate_prepare_paths(args: argparse.Namespace,
                                conf_obj: conf.Config) -> AttributeDict:
    import os
    
    output_path = f"{conf_obj.paths.output}/{args.network}/test/{args.iteration}"
    evaluation_path = f"{conf_obj.paths.evaluation}/{args.network}"
    result_file = f"{evaluation_path}/results-{args.iteration}"
    label_file = f"{output_path}/labels.bin"
    filenames_file = f"{output_path}/filenames.bin"
    predictions_file = f"{output_path}/predictions"
    predictions_per_class_file = f"{output_path}/predictions_class"
    predictions_glob_string = f"{output_path}/*ssd_predictions_transformed*"
    label_glob_string = f"{output_path}/*ssd_label*"
    
    os.makedirs(evaluation_path, exist_ok=True)
    
    return AttributeDict({
        "output_path": output_path,
        "evaluation_path": evaluation_path,
        "result_file": result_file,
        "label_file": label_file,
        "filenames_file": filenames_file,
        "predictions_file": predictions_file,
        "predictions_per_class_file": predictions_per_class_file,
        "predictions_glob_string": predictions_glob_string,
        "label_glob_string": label_glob_string
    })


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
                                     conf_obj: conf.Config) -> Tuple[str, str]:
    import os
    
    metrics_file = f"{conf_obj.paths.evaluation}/{args.network}/results-{args.iteration}"
    output_path = f"{conf_obj.paths.output}/{args.network}/visualise/{args.iteration}"
    
    os.makedirs(output_path, exist_ok=True)
    
    return output_path, metrics_file


def _visualise_all_prepare_paths(args: argparse.Namespace,
                                 conf_obj: conf.Config) -> Tuple[str, List[str]]:
    import os
    
    metrics_files = [
        f"{conf_obj.paths.evaluation}/ssd/results-{args.vanilla_ssd_001_iteration}",
        f"{conf_obj.paths.evaluation}/ssd/results-{args.vanilla_ssd_02_iteration}",
        f"{conf_obj.paths.evaluation}/ssd/results-{args.entropy_ssd_001_iteration}",
        f"{conf_obj.paths.evaluation}/bayesian_ssd/results-{args.bayesian_ssd_no_do_no_nms_iteration}",
        f"{conf_obj.paths.evaluation}/bayesian_ssd/results-{args.bayesian_ssd_no_do_iteration}",
        f"{conf_obj.paths.evaluation}/bayesian_ssd/results-{args.bayesian_ssd_do_09_iteration}",
        f"{conf_obj.paths.evaluation}/bayesian_ssd/results-{args.bayesian_ssd_do_05_iteration}"
    ]
    output_path = f"{conf_obj.paths.output}/all/visualise/"
    
    os.makedirs(output_path, exist_ok=True)
    
    return output_path, metrics_files


def _ssd_train_load_gt(conf_obj: conf.Config) -> AttributeDict:

    import pickle

    with open(f"{conf_obj.paths.scenenet_gt_train}/photo_paths.bin", "rb") as file:
        file_names_train = pickle.load(file)
    with open(f"{conf_obj.paths.scenenet_gt_train}/instances.bin", "rb") as file:
        instances_train = pickle.load(file)
    with open(f"{conf_obj.paths.scenenet_gt_val}/photo_paths.bin", "rb") as file:
        file_names_val = pickle.load(file)
    with open(f"{conf_obj.paths.scenenet_gt_val}/instances.bin", "rb") as file:
        instances_val = pickle.load(file)
        
    return AttributeDict({
        "file_names_train": file_names_train,
        "instances_train": instances_train,
        "file_names_val": file_names_val,
        "instances_val": instances_val
    })


def _ssd_test_load_gt(conf_obj: conf.Config) -> AttributeDict:
    import pickle
    
    with open(f"{conf_obj.paths.scenenet_gt_test}/photo_paths.bin", "rb") as file:
        file_names = pickle.load(file)
    with open(f"{conf_obj.paths.scenenet_gt_test}/instances.bin", "rb") as file:
        instances = pickle.load(file)
        
    return AttributeDict({
        "file_names": file_names,
        "instances": instances
    })


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
                              conf_obj: conf.Config,
                              load_data: callable,
                              gt: AttributeDict,
                              predictor_sizes: Sequence[Sequence[int]]) -> AttributeDict:
    nr_trajectories = conf_obj.parameters.nr_trajectories if conf_obj.parameters.nr_trajectories != -1 else None
        
    train_generator, train_length, train_debug_generator = \
        load_data(gt.file_names_train, gt.instances_train, conf_obj.paths.coco,
                  predictor_sizes=predictor_sizes,
                  batch_size=conf_obj.parameters.batch_size,
                  image_size=conf_obj.parameters.ssd_image_size,
                  training=True, evaluation=False, augment=False,
                  debug=args.debug,
                  nr_trajectories=nr_trajectories)
    
    val_generator, val_length, val_debug_generator = \
        load_data(gt.file_names_val, gt.instances_val, conf_obj.paths.coco,
                  predictor_sizes=predictor_sizes,
                  batch_size=conf_obj.parameters.batch_size,
                  image_size=conf_obj.parameters.ssd_image_size,
                  training=False, evaluation=False, augment=False,
                  debug=args.debug,
                  nr_trajectories=nr_trajectories)
    
    return AttributeDict({
        "train_generator": train_generator,
        "train_length": train_length,
        "train_debug_generator": train_debug_generator,
        "val_generator": val_generator,
        "val_length": val_length,
        "val_debug_generator": val_debug_generator
    })


def _ssd_test_get_generators(args: argparse.Namespace,
                             conf_obj: conf.Config,
                             load_data_coco: callable,
                             load_data_scenenet: callable,
                             gt: AttributeDict,
                             predictor_sizes: Sequence[Sequence[int]]) -> AttributeDict:
    
    from twomartens.masterthesis import data
    
    nr_trajectories = conf_obj.parameters.nr_trajectories if conf_obj.parameters.nr_trajectories != -1 else None
    
    if conf_obj.parameters.ssd_use_coco:
        generator, length, debug_generator = load_data_coco(data.clean_dataset,
                                                            data.group_bboxes_to_images,
                                                            conf_obj.paths.coco,
                                                            conf_obj.parameters.batch_size,
                                                            conf_obj.parameters.ssd_image_size,
                                                            training=False, evaluation=True, augment=False,
                                                            debug=args.debug,
                                                            predictor_sizes=predictor_sizes)
    else:
        generator, length, debug_generator = load_data_scenenet(gt.file_names, gt.instances, conf_obj.paths.coco,
                                                                predictor_sizes=predictor_sizes,
                                                                batch_size=conf_obj.parameters.batch_size,
                                                                image_size=conf_obj.parameters.ssd_image_size,
                                                                training=False, evaluation=True, augment=False,
                                                                debug=args.debug,
                                                                nr_trajectories=nr_trajectories)
    
    return AttributeDict({
        "generator": generator,
        "length": length,
        "debug_generator": debug_generator
    })


def _ssd_debug_save_images(args: argparse.Namespace, conf_obj: conf.Config,
                           paths: AttributeDict,
                           save_images: callable, get_coco_cat_maps_func: callable,
                           train_generator: Generator) -> None:
    
    if args.debug and conf_obj.debug.train_images:
        train_data = next(train_generator)
        train_images = train_data[0]
        train_labels = train_data[1]
        train_labels_not_encoded = train_data[2]
        
        save_images(train_images, train_labels_not_encoded,
                    paths.summary_path, conf_obj.paths.coco, conf_obj.parameters.ssd_image_size,
                    get_coco_cat_maps_func, "before-encoding")

        save_images(train_images, train_labels,
                    paths.summary_path, conf_obj.paths.coco, conf_obj.parameters.ssd_image_size,
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


def _ssd_save_history(summary_path: str, history: tf.keras.callbacks.History) -> None:
    import pickle
    
    with open(f"{summary_path}/history", "wb") as file:
        pickle.dump(history.history, file)


def _ssd_evaluate_get_results(**kwargs) -> Dict[str, Union[np.ndarray, float, int]]:
    results = {}
    for key in kwargs:
        results[key] = kwargs[key]
    
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


def _visualise_precision_recall_all(precision: Sequence[np.ndarray],
                                    recall: Sequence[np.ndarray],
                                    legend: Sequence[str],
                                    output_path: str, file_suffix: str) -> None:
    from matplotlib import pyplot
    
    figure = pyplot.figure()
    
    pyplot.ylabel("precision")
    pyplot.xlabel("recall")
    
    for prec, rec in zip(precision, recall):
        pyplot.plot(rec, prec)
    
    pyplot.legend(legend, loc="upper right")
    pyplot.xlim(xmax=1)
    pyplot.savefig(f"{output_path}/precision-recall-all-{file_suffix}.png")
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


def _visualise_ose_f1_all(open_set_error: Sequence[np.ndarray],
                          f1_scores: Sequence[np.ndarray],
                          legend: Sequence[str],
                          output_path: str, file_suffix: str) -> None:
    from matplotlib import pyplot
    
    figure = pyplot.figure()
    axis = pyplot.gca()
    axis.set_autoscale_on(False)
    pyplot.ylabel("absolute ose")
    pyplot.xlabel("f1 score")
    
    for ose, f1 in zip(open_set_error, f1_scores):
        pyplot.plot(f1, ose)
    
    axis.axis([0, 0.4, 0, 8000])
    pyplot.legend(legend, loc="upper right")
    
    pyplot.savefig(f"{output_path}/ose-f1-all-{file_suffix}.png")
    pyplot.close(figure)
