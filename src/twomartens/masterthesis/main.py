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

from twomartens.masterthesis import cli


def main() -> None:
    """
    Provides command line interface.
    """
    parser = argparse.ArgumentParser(
        description="Train, test, and use SSD with novelty detection.",
    )
    
    parser.add_argument("--verbose", action="store_true", help="provide to get extra output")
    parser.add_argument("--debug", action="store_true", help="provide to collect tensorboard summaries")
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
    _build_test(test_parser)
    
    args = parser.parse_args()
    
    if args.action == "train":
        cli.train(args)
    elif args.action == "test":
        cli.test(args)
    elif args.action == "val":
        cli.val(args)
    elif args.action == "prepare":
        cli.prepare(args)


def _build_prepare(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("scenenet_path", type=str, help="the path to the SceneNet RGB-D data set")
    parser.add_argument("protobuf_path", type=str, help="the path to the SceneNet RGB-D protobuf file")
    parser.add_argument("ground_truth_path", type=str, help="the path where the ground truth should be stored")
    

def _build_train(parser: argparse.ArgumentParser) -> None:
    sub_parsers = parser.add_subparsers(dest="network")
    sub_parsers.required = True
    
    ssd_parser = sub_parsers.add_parser("ssd", help="SSD")
    # ssd_bayesian_parser = sub_parsers.add_parser("bayesian_ssd", help="SSD with dropout layers")
    auto_encoder_parser = sub_parsers.add_parser("auto_encoder", help="Auto-encoder network")
    
    # build sub parsers
    _build_ssd_train(ssd_parser)
    # _build_bayesian_ssd(ssd_bayesian_parser)
    _build_auto_encoder_train(auto_encoder_parser)
 

def _build_ssd_train(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--coco_path", type=str, help="the path to the COCO data set")
    parser.add_argument("--weights_path", type=str, help="path to the weights directory")
    parser.add_argument("--ground_truth_path_train", type=str,
                        help="path to the prepared ground truth directory for training")
    parser.add_argument("--ground_truth_path_val", type=str,
                        help="path to the prepared ground truth directory for validation")
    parser.add_argument("--summary_path", type=str, help="path to the summaries directory")
    parser.add_argument("num_epochs", type=int, help="the number of epochs to train", default=80)
    parser.add_argument("iteration", type=int, help="the training iteration")
    

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
    parser.add_argument("train_iteration", type=int, help="the train iteration")
    

def _build_auto_encoder_val(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--coco_path", type=str, help="the path to the COCO data set")
    parser.add_argument("--weights_path", type=str, help="path to the weights directory")
    parser.add_argument("--summary_path", type=str, help="path to the summaries directory")
    parser.add_argument("category", type=int, help="the COCO category to validate")
    parser.add_argument("category_trained", type=int, help="the trained COCO category")
    parser.add_argument("iteration", type=int, help="the validation iteration")
    parser.add_argument("iteration_trained", type=int, help="the training iteration")


def _build_test(parser: argparse.ArgumentParser) -> None:
    sub_parsers = parser.add_subparsers(dest="network")
    sub_parsers.required = True

    ssd_bayesian_parser = sub_parsers.add_parser("bayesian_ssd", help="SSD with dropout layers")
    ssd_parser = sub_parsers.add_parser("ssd", help="SSD")

    # build sub parsers
    _build_ssd_test(ssd_bayesian_parser)
    _build_ssd_test(ssd_parser)


def _build_ssd_test(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output_path", type=str, help="path to the output directory")
    parser.add_argument("--evaluation_path", type=str, help="path to the directory for the evaluation results")
    parser.add_argument("iteration", type=int, help="the validation iteration to use")


if __name__ == "__main__":
    main()
