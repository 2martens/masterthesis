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
    
    train_parser = sub_parsers.add_parser("train", help="Train a network")
    test_parser = sub_parsers.add_parser("test", help="Test a network")
    val_parser = sub_parsers.add_parser("val", help="Validate a network")
    
    # build sub parsers
    _build_train(train_parser)
    _build_val(val_parser)
    
    args = parser.parse_args()
    
    if args.action == "train":
        _train(args)
    elif args.action == "test":
        _test(args)
    elif args.action == "val":
        _val(args)


def _build_train(parser: argparse.ArgumentParser) -> None:
    sub_parsers = parser.add_subparsers(dest="network")
    sub_parsers.required = True
    
    # ssd_bayesian_parser = sub_parsers.add_parser("bayesian_ssd", help="SSD with dropout layers")
    auto_encoder_parser = sub_parsers.add_parser("auto_encoder", help="Auto-encoder network")
    
    # build sub parsers
    # _build_bayesian_ssd(ssd_bayesian_parser)
    _build_auto_encoder_train(auto_encoder_parser)
    

def _build_val(parser: argparse.ArgumentParser) -> None:
    sub_parsers = parser.add_subparsers(dest="network")
    sub_parsers.required = True
    
    # ssd_bayesian_parser = sub_parsers.add_parser("bayesian_ssd", help="SSD with dropout layers")
    auto_encoder_parser = sub_parsers.add_parser("auto_encoder", help="Auto-encoder network")
    
    # build sub parsers
    _build_auto_encoder_val(auto_encoder_parser)


def _build_auto_encoder_train(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--coco_path", type=str, help="the path to the COCO data set")
    parser.add_argument("--weights_path", type=str, help="path to the weights directory")
    parser.add_argument("--summary_path", type=str, help="path to the summaries directory")
    parser.add_argument("category", type=int, help="the COCO category to use")
    parser.add_argument("num_epochs", type=int, help="the number of epochs to train", default=80)
    parser.add_argument("iteration", type=int, help="the training iteration")


def _build_auto_encoder_val(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--coco_path", type=str, help="the path to the COCO data set")
    parser.add_argument("--weights_path", type=str, help="path to the weights directory")
    parser.add_argument("--summary_path", type=str, help="path to the summaries directory")
    parser.add_argument("category", type=int, help="the COCO category to validate")
    parser.add_argument("category_trained", type=int, help="the trained COCO category")
    parser.add_argument("iteration", type=int, help="the validation iteration")
    parser.add_argument("iteration_trained", type=int, help="the training iteration")


def _build_bayesian_ssd(parser: argparse.ArgumentParser) -> None:
    raise NotImplementedError


def _train(args: argparse.Namespace) -> None:
    if args.network == "bayesian_ssd":
        _bayesian_ssd_train(args)
    elif args.network == "auto_encoder":
        _auto_encoder_train(args)


def _test(args: argparse.Namespace) -> None:
    raise NotImplementedError


def _val(args: argparse.Namespace) -> None:
    from twomartens.masterthesis import data
    from twomartens.masterthesis.aae import run
    import tensorflow as tf
    from tensorflow.python.ops import summary_ops_v2
    
    tf.enable_eager_execution()
    coco_path = args.coco_path
    category = args.category
    category_trained = args.category_trained
    batch_size = 16
    coco_data = data.load_coco_val(coco_path, category, num_epochs=1,
                                   batch_size=batch_size, resized_shape=(256, 256))
    use_summary_writer = summary_ops_v2.create_file_writer(
        f"{args.summary_path}/val/category-{category}/{args.iteration}"
    )
    if args.debug:
        with use_summary_writer.as_default():
            run.run_simple(coco_data, iteration=args.iteration_trained,
                           weights_prefix=f"{args.weights_path}/category-{category_trained}",
                           zsize=16, verbose=args.verbose, channels=3, batch_size=batch_size)
    else:
        run.run_simple(coco_data, iteration=args.iteration_trained,
                       weights_prefix=f"{args.weights_path}/category-{category_trained}",
                       zsize=16, verbose=args.verbose, channels=3, batch_size=batch_size)


def _auto_encoder_train(args: argparse.Namespace) -> None:
    from twomartens.masterthesis import data
    from twomartens.masterthesis.aae import train
    import tensorflow as tf
    from tensorflow.python.ops import summary_ops_v2

    tf.enable_eager_execution()
    coco_path = args.coco_path
    category = args.category
    batch_size = 16
    coco_data = data.load_coco_train(coco_path, category, num_epochs=args.num_epochs, batch_size=batch_size,
                                     resized_shape=(256, 256))
    train_summary_writer = summary_ops_v2.create_file_writer(
        f"{args.summary_path}/train/category-{category}/{args.iteration}"
    )
    if args.debug:
        with train_summary_writer.as_default():
            train.train_simple(coco_data, iteration=args.iteration,
                               weights_prefix=f"{args.weights_path}/category-{category}",
                               zsize=16, lr=0.0001, verbose=args.verbose,
                               channels=3, train_epoch=args.num_epochs, batch_size=batch_size)
    else:
        train.train_simple(coco_data, iteration=args.iteration,
                           weights_prefix=f"{args.weights_path}/category-{category}",
                           zsize=16, lr=0.0001, verbose=args.verbose,
                           channels=3, train_epoch=args.num_epochs, batch_size=batch_size)


def _bayesian_ssd_train(args: argparse.Namespace) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
