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
    train(...): trains a network
    test(...): evaluates a network
    val(...): runs predictions on the validation data
    prepare(...): prepares the SceneNet ground truth data
"""
import argparse


def train(args: argparse.Namespace) -> None:
    if args.network == "ssd" or args.network == "bayesian_ssd":
        _ssd_train(args)
    elif args.network == "auto_encoder":
        _auto_encoder_train(args)


def _ssd_train(args: argparse.Namespace) -> None:
    import os
    import pickle
    
    import tensorflow as tf
    from tensorflow.python.ops import summary_ops_v2

    from twomartens.masterthesis import data
    from twomartens.masterthesis import ssd
    
    tf.enable_eager_execution()
    
    batch_size = 16
    image_size = 300
    use_dropout = False if args.network == "ssd" else True
    
    pre_trained_weights_file = f"{args.weights_path}/VGG_coco_SSD_300x300_iter_400000.h5"
    weights_path = f"{args.weights_path}/train/{args.network}/"
    os.makedirs(weights_path, exist_ok=True)
    
    # load prepared ground truth
    with open(f"{args.ground_truth_path}/photo_paths.bin", "rb") as file:
        file_names_photos = pickle.load(file)
    with open(f"{args.ground_truth_path}/instances.bin", "rb") as file:
        instances = pickle.load(file)
    
    scenenet_data, nr_digits, length_dataset = \
        data.load_scenenet_data(file_names_photos, instances, args.coco_path,
                                batch_size=batch_size,
                                resized_shape=(image_size, image_size),
                                mode="training")
    del file_names_photos, instances

    use_summary_writer = summary_ops_v2.create_file_writer(
        f"{args.summary_path}/train/{args.network}/{args.iteration}"
    )
    
    if args.debug:
        with use_summary_writer.as_default():
            ssd.train(scenenet_data, args.iteration, use_dropout, length_dataset,
                      weights_prefix=weights_path,
                      weights_path=pre_trained_weights_file, batch_size=batch_size,
                      nr_epochs=args.num_epochs,
                      verbose=args.verbose)
    else:
        ssd.train(scenenet_data, args.iteration, use_dropout, length_dataset,
                  weights_prefix=weights_path,
                  weights_path=pre_trained_weights_file, batch_size=batch_size,
                  nr_epochs=args.num_epochs,
                  verbose=args.verbose)


def _auto_encoder_train(args: argparse.Namespace) -> None:
    from twomartens.masterthesis import data
    from twomartens.masterthesis.aae import train
    import tensorflow as tf
    from tensorflow.python.ops import summary_ops_v2
    
    tf.enable_eager_execution()
    coco_path = args.coco_path
    category = args.category
    batch_size = 16
    image_size = 256
    coco_data = data.load_coco_train(coco_path, category, num_epochs=args.num_epochs, batch_size=batch_size,
                                     resized_shape=(image_size, image_size))
    train_summary_writer = summary_ops_v2.create_file_writer(
        f"{args.summary_path}/train/category-{category}/{args.iteration}"
    )
    if args.debug:
        with train_summary_writer.as_default():
            train.train_simple(coco_data, iteration=args.iteration,
                               weights_prefix=f"{args.weights_path}/category-{category}",
                               zsize=16, lr=0.0001, verbose=args.verbose, image_size=image_size,
                               channels=3, train_epoch=args.num_epochs, batch_size=batch_size)
    else:
        train.train_simple(coco_data, iteration=args.iteration,
                           weights_prefix=f"{args.weights_path}/category-{category}",
                           zsize=16, lr=0.0001, verbose=args.verbose, image_size=image_size,
                           channels=3, train_epoch=args.num_epochs, batch_size=batch_size)


def test(args: argparse.Namespace) -> None:
    if args.network == "ssd":
        _ssd_test(args)
    else:
        raise NotImplementedError


def _ssd_test(args: argparse.Namespace) -> None:
    import glob
    import os
    import pickle
    
    import numpy as np
    import tensorflow as tf
    
    from twomartens.masterthesis import evaluate
    from twomartens.masterthesis import ssd
    
    tf.enable_eager_execution()
    
    batch_size = 16
    use_dropout = False if args.network == "ssd" else True
    output_path = f"{args.output_path}/val/{args.network}/{args.iteration}"
    evaluation_path = f"{args.evaluation_path}/{args.network}"
    result_file = f"{evaluation_path}/results-{args.iteration}.bin"
    label_file = f"{output_path}/labels.bin"
    predictions_file = f"{output_path}/predictions.bin"
    predictions_per_class_file = f"{output_path}/predictions_class.bin"
    os.makedirs(evaluation_path, exist_ok=True)
    
    # retrieve labels and un-batch them
    files = glob.glob(f"{output_path}/*ssd_labels*")
    labels = []
    for filename in files:
        with open(filename, "rb") as file:
            # get labels per batch
            _labels = np.asarray(pickle.load(file))
            # exclude padded label entries
            for i in range(_labels.shape[0]):
                image_labels = _labels[i]
                image_labels = image_labels[image_labels[:, 0] != -1]
                labels.append(image_labels)
    # store labels for later use
    with open(label_file, "wb") as file:
        pickle.dump(labels, file)
    
    number_gt_per_class = evaluate.get_number_gt_per_class(labels, ssd.N_CLASSES)
    
    # retrieve predictions and un-batch them
    files = glob.glob(f"{output_path}/*ssd_predictions*")
    predictions = []
    for filename in files:
        with open(filename, "rb") as file:
            # get predictions per batch
            _predictions = pickle.load(file)
            predictions.extend(_predictions)
    del _predictions
    
    # prepare predictions for further use
    with open(predictions_file, "wb") as file:
        pickle.dump(predictions, file)
    
    predictions_per_class = evaluate.prepare_predictions(predictions, ssd.N_CLASSES)
    del predictions
    
    with open(predictions_per_class_file, "wb") as file:
        pickle.dump(predictions_per_class, file)
    
    # compute matches between predictions and ground truth
    true_positives, false_positives, \
    cum_true_positives, cum_false_positives, open_set_error = evaluate.match_predictions(predictions_per_class,
                                                                                         labels,
                                                                                         ssd.N_CLASSES)
    del labels
    cum_precisions, cum_recalls = evaluate.get_precision_recall(number_gt_per_class,
                                                                cum_true_positives,
                                                                cum_false_positives,
                                                                ssd.N_CLASSES)
    f1_scores = evaluate.get_f1_score(cum_precisions, cum_recalls, ssd.N_CLASSES)
    average_precisions = evaluate.get_mean_average_precisions(cum_precisions, cum_recalls, ssd.N_CLASSES)
    mean_average_precision = evaluate.get_mean_average_precision(average_precisions)
    
    results = {
        "true_positives":             true_positives,
        "false_positives":            false_positives,
        "cumulative_true_positives":  cum_true_positives,
        "cumulative_false_positives": cum_false_positives,
        "cumulative_precisions":      cum_precisions,
        "cumulative_recalls":         cum_recalls,
        "f1_scores":                  f1_scores,
        "mean_average_precisions":    average_precisions,
        "mean_average_precision":     mean_average_precision,
        "open_set_error":             open_set_error
    }
    
    with open(result_file, "wb") as file:
        pickle.dump(results, file)


def val(args: argparse.Namespace) -> None:
    if args.network == "ssd" or args.network == "bayesian_ssd":
        _ssd_val(args)
    elif args.network == "auto_encoder":
        _auto_encoder_val(args)


def _ssd_val(args: argparse.Namespace) -> None:
    import pickle
    import os
    
    import tensorflow as tf
    from tensorflow.python.ops import summary_ops_v2
    
    from twomartens.masterthesis import data
    from twomartens.masterthesis import ssd
    
    config = tf.ConfigProto()
    config.log_device_placement = False
    config.gpu_options.allow_growth = False
    tf.enable_eager_execution(config=config)
    
    batch_size = 16
    image_size = 300
    forward_passes_per_image = 10
    use_dropout = False if args.network == "ssd" else True
    
    weights_file = f"{args.weights_path}/VGG_coco_SSD_300x300_iter_400000.h5"
    checkpoint_path = f"{args.weights_path}/train/{args.network}/{args.train_iteration}"
    output_path = f"{args.output_path}/val/{args.network}/{args.iteration}/"
    os.makedirs(output_path, exist_ok=True)
    
    # load prepared ground truth
    with open(f"{args.ground_truth_path}/photo_paths.bin", "rb") as file:
        file_names_photos = pickle.load(file)
    with open(f"{args.ground_truth_path}/instances.bin", "rb") as file:
        instances = pickle.load(file)
    
    scenenet_data, nr_digits, length_dataset = \
        data.load_scenenet_data(file_names_photos, instances, args.coco_path,
                                batch_size=batch_size,
                                resized_shape=(image_size, image_size))
    del file_names_photos, instances
    
    use_summary_writer = summary_ops_v2.create_file_writer(
        f"{args.summary_path}/val/{args.network}/{args.iteration}"
    )
    if args.debug:
        with use_summary_writer.as_default():
            ssd.predict(scenenet_data, use_dropout, output_path, weights_file, checkpoint_path,
                        nr_digits=nr_digits, forward_passes_per_image=forward_passes_per_image)
    else:
        ssd.predict(scenenet_data, use_dropout, output_path, weights_file, checkpoint_path,
                    nr_digits=nr_digits, forward_passes_per_image=forward_passes_per_image)


def _auto_encoder_val(args: argparse.Namespace) -> None:
    from twomartens.masterthesis import data
    from twomartens.masterthesis.aae import run
    import tensorflow as tf
    from tensorflow.python.ops import summary_ops_v2
    
    tf.enable_eager_execution()
    coco_path = args.coco_path
    category = args.category
    category_trained = args.category_trained
    batch_size = 16
    image_size = 256
    coco_data = data.load_coco_val(coco_path, category, num_epochs=1,
                                   batch_size=batch_size, resized_shape=(image_size, image_size))
    use_summary_writer = summary_ops_v2.create_file_writer(
        f"{args.summary_path}/val/category-{category}/{args.iteration}"
    )
    if args.debug:
        with use_summary_writer.as_default():
            run.run_simple(coco_data, iteration=args.iteration_trained,
                           weights_prefix=f"{args.weights_path}/category-{category_trained}",
                           zsize=16, verbose=args.verbose, channels=3, batch_size=batch_size,
                           image_size=image_size)
    else:
        run.run_simple(coco_data, iteration=args.iteration_trained,
                       weights_prefix=f"{args.weights_path}/category-{category_trained}",
                       zsize=16, verbose=args.verbose, channels=3, batch_size=batch_size,
                       image_size=image_size)


def prepare(args: argparse.Namespace) -> None:
    import pickle
    
    from twomartens.masterthesis import data
    
    file_names_photos, file_names_instances, instances = data.prepare_scenenet_val(args.scenenet_path,
                                                                                   args.protobuf_path)
    with open(f"{args.ground_truth_path}/photo_paths.bin", "wb") as file:
        pickle.dump(file_names_photos, file)
    with open(f"{args.ground_truth_path}/instance_paths.bin", "wb") as file:
        pickle.dump(file_names_instances, file)
    with open(f"{args.ground_truth_path}/instances.bin", "wb") as file:
        pickle.dump(instances, file)
