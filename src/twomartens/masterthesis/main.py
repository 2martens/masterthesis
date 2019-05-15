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
Provides entry point into the application.

Functions:
    main(...): provides command line interface
"""
import argparse

def main() -> None:
    """
    Provides command line interface.
    """
    parser = argparse.ArgumentParser(
        description="Train, test, and use SSD with novelty detection.",
    )
    
    parser.add_argument("--verbose", default=False, action="store_true", help="provide to get extra output")
    parser.add_argument("--debug", default=False, action="store_true", help="provide to collect tensorboard summaries")
    parser.add_argument('--version', action='version', version='2martens Masterthesis 0.1.0')
    sub_parsers = parser.add_subparsers(dest="action")
    sub_parsers.required = True
    
    prepare_parser = sub_parsers.add_parser("prepare", help="Prepare SceneNet RGB-D ground truth")
    train_parser = sub_parsers.add_parser("train", help="Train a network")
    test_parser = sub_parsers.add_parser("test", help="Test a network")
    val_parser = sub_parsers.add_parser("val", help="Validate a network")
    
    # build sub parsers
    _build_prepare(prepare_parser)
    _build_train(train_parser)
    _build_val(val_parser)
    
    args = parser.parse_args()
    
    if args.action == "train":
        _train(args)
    elif args.action == "test":
        _test(args)
    elif args.action == "val":
        _val(args)
    elif args.action == "prepare":
        _prepare(args)


def _build_prepare(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("scenenet_path", type=str, help="the path to the SceneNet RGB-D validation data set")
    parser.add_argument("protobuf_path", type=str, help="the path to the SceneNet RGB-D validation protobuf file")
    parser.add_argument("ground_truth_path", type=str, help="the path where the ground truth should be stored")
    

def _build_train(parser: argparse.ArgumentParser) -> None:
    sub_parsers = parser.add_subparsers(dest="network")
    sub_parsers.required = True
    
    # ssd_bayesian_parser = sub_parsers.add_parser("bayesian_ssd", help="SSD with dropout layers")
    auto_encoder_parser = sub_parsers.add_parser("auto_encoder", help="Auto-encoder network")
    
    # build sub parsers
    # _build_bayesian_ssd(ssd_bayesian_parser)
    _build_auto_encoder_train(auto_encoder_parser)
    

def _build_auto_encoder_train(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--coco_path", type=str, help="the path to the COCO data set")
    parser.add_argument("--weights_path", type=str, help="path to the weights directory")
    parser.add_argument("--summary_path", type=str, help="path to the summaries directory")
    parser.add_argument("category", type=int, help="the COCO category to use")
    parser.add_argument("num_epochs", type=int, help="the number of epochs to train", default=80)
    parser.add_argument("iteration", type=int, help="the training iteration")
    

def _build_val(parser: argparse.ArgumentParser) -> None:
    sub_parsers = parser.add_subparsers(dest="network")
    sub_parsers.required = True
    
    ssd_bayesian_parser = sub_parsers.add_parser("bayesian_ssd", help="SSD with dropout layers")
    ssd_parser = sub_parsers.add_parser("ssd", help="SSD")
    auto_encoder_parser = sub_parsers.add_parser("auto_encoder", help="Auto-encoder network")
    
    # build sub parsers
    _build_ssd_val(ssd_bayesian_parser)
    _build_ssd_val(ssd_parser)
    _build_auto_encoder_val(auto_encoder_parser)


def _build_ssd_val(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--coco_path", type=str, help="the path to the COCO data set")
    parser.add_argument("--weights_path", type=str, help="path to the weights directory")
    parser.add_argument("--ground_truth_path", type=str, help="path to the prepared ground truth directory")
    parser.add_argument("--summary_path", type=str, help="path to the summaries directory")
    parser.add_argument("--output_path", type=str, help="path to the output directory")
    parser.add_argument("iteration", type=int, help="the validation iteration")
    

def _build_auto_encoder_val(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--coco_path", type=str, help="the path to the COCO data set")
    parser.add_argument("--weights_path", type=str, help="path to the weights directory")
    parser.add_argument("--summary_path", type=str, help="path to the summaries directory")
    parser.add_argument("category", type=int, help="the COCO category to validate")
    parser.add_argument("category_trained", type=int, help="the trained COCO category")
    parser.add_argument("iteration", type=int, help="the validation iteration")
    parser.add_argument("iteration_trained", type=int, help="the training iteration")


def _train(args: argparse.Namespace) -> None:
    if args.network == "auto_encoder":
        _auto_encoder_train(args)


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


def _test(args: argparse.Namespace) -> None:
    if args.network == "ssd":
        _ssd_test(args)
    else:
        raise NotImplementedError


def _ssd_test(args: argparse.Namespace) -> None:
    import glob
    import pickle
    
    import numpy as np
    import tensorflow as tf
    
    tf.enable_eager_execution()
    
    batch_size = 16
    use_dropout = False if args.network == "ssd" else True
    output_path = f"{args.output_path}/val/{args.network}/{args.iteration}"
    label_file = f"{output_path}/labels.bin"
    
    # retrieve labels and un-batch them
    files = glob.glob(f"{output_path}/*ssd_labels*")
    labels = []
    for filename in files:
        with open(filename, "rb") as file:
            # get labels per batch
            _labels = pickle.load(file)
            # exclude padded label entries
            real_labels = _labels[:, :, 0] != -1
            labels.extend(_labels[real_labels])
    # store labels for later use
    with open(label_file, "wb") as file:
        pickle.dump(labels, file)
    
    # TODO implement evaluate.py analogous to average_precision_evaluator


def _val(args: argparse.Namespace) -> None:
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
    forward_passes_per_image = 42
    use_dropout = False if args.network == "ssd" else True
    
    weights_file = f"{args.weights_path}/VGG_coco_SSD_300x300_iter_400000.h5"
    output_path = f"{args.output_path}/val/{args.network}/{args.iteration}/"
    os.makedirs(output_path, exist_ok=True)
    
    # load prepared ground truth
    with open(f"{args.ground_truth_path}/photo_paths.bin", "rb") as file:
        file_names_photos = pickle.load(file)
    with open(f"{args.ground_truth_path}/instances.bin", "rb") as file:
        instances = pickle.load(file)
    
    scenenet_data, nr_digits = data.load_scenenet_val(file_names_photos, instances, args.coco_path,
                                                      batch_size=batch_size,
                                                      resized_shape=(image_size, image_size))
    del file_names_photos, instances
    
    use_summary_writer = summary_ops_v2.create_file_writer(
        f"{args.summary_path}/val/{args.network}/{args.iteration}"
    )
    if args.debug:
        with use_summary_writer.as_default():
            ssd.predict(scenenet_data, use_dropout, output_path, weights_file, nr_digits=nr_digits,
                        forward_passes_per_image=forward_passes_per_image)
    else:
        ssd.predict(scenenet_data, use_dropout, output_path, weights_file, nr_digits=nr_digits,
                    forward_passes_per_image=forward_passes_per_image)


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


def _prepare(args: argparse.Namespace) -> None:
    import pickle
    
    from twomartens.masterthesis import data
    
    file_names_photos, file_names_instances, instances = data.prepare_scenenet_val(args.scenenet_path, args.protobuf_path)
    with open(f"{args.ground_truth_path}/photo_paths.bin", "wb") as file:
        pickle.dump(file_names_photos, file)
    with open(f"{args.ground_truth_path}/instance_paths.bin", "wb") as file:
        pickle.dump(file_names_instances, file)
    with open(f"{args.ground_truth_path}/instances.bin", "wb") as file:
        pickle.dump(instances, file)


if __name__ == "__main__":
    main()
