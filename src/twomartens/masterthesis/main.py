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
from typing import List

from twomartens.masterthesis import cli


def main() -> None:
    """
    Provides command line interface.
    """
    parser = argparse.ArgumentParser(
        description="Train, test, and use SSD with novelty detection.",
    )
    
    _build_general(parser)
    sub_parsers = _build_sub_parsers(parser)
    
    _build_config(sub_parsers[0])
    _build_prepare(sub_parsers[1])
    _build_train(sub_parsers[2])
    _build_test(sub_parsers[3])
    _build_evaluate(sub_parsers[4])
    _build_visualise(sub_parsers[5])
    _build_visualise_metrics(sub_parsers[6])
    _build_visualise_all(sub_parsers[7])
    _build_measure(sub_parsers[8])
    
    args = _get_user_input(parser)
    _execute_action(args)


def _build_general(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--verbose", action="store_true", help="provide to get extra output")
    parser.add_argument("--debug", action="store_true", help="activate debug functionality")
    parser.add_argument('--version', action='version', version='2martens Masterthesis 0.1.0')


def _build_sub_parsers(parser: argparse.ArgumentParser) -> List[argparse.ArgumentParser]:
    sub_parsers = parser.add_subparsers(dest="component")
    sub_parsers.required = True
    config_parser = sub_parsers.add_parser("config", help="Get and set config values")
    prepare_parser = sub_parsers.add_parser("prepare", help="Prepare SceneNet RGB-D ground truth")
    train_parser = sub_parsers.add_parser("train", help="Train a network")
    test_parser = sub_parsers.add_parser("test", help="Test a network")
    evaluate_parser = sub_parsers.add_parser("evaluate", help="Evaluate a network")
    visualise_parser = sub_parsers.add_parser("visualise", help="Visualise the ground truth")
    visualise_metrics_parser = sub_parsers.add_parser("visualise_metrics", help="Visualise the evaluation results")
    visualise_all_parser = sub_parsers.add_parser("visualise_all",
                                                  help="Visualise evaluation results of multiple runs ")
    measure_parser = sub_parsers.add_parser("measure_mapping", help="Measure the number of instances per COCO category")
    
    return [
        config_parser,
        prepare_parser,
        train_parser,
        test_parser,
        evaluate_parser,
        visualise_parser,
        visualise_metrics_parser,
        visualise_all_parser,
        measure_parser
    ]


def _get_user_input(parser: argparse.ArgumentParser) -> argparse.Namespace:
    return parser.parse_args()


def _get_action(component: str) -> callable:
    return getattr(cli, component)


def _execute_action(args: argparse.Namespace) -> None:
    getattr(cli, args.component)(args)


def _build_config(parser: argparse.ArgumentParser) -> None:
    sub_parsers = parser.add_subparsers(dest="action")
    sub_parsers.required = True
    
    get_parser = sub_parsers.add_parser("get", help="Get a config value")
    set_parser = sub_parsers.add_parser("set", help="Set a config value")
    sub_parsers.add_parser("list", help="List all config values")
    
    _build_config_get(get_parser)
    _build_config_set(set_parser)


def _build_config_get(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("property", type=str, help="config property to retrieve")


def _build_config_set(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("property", type=str, help="config property to set")
    parser.add_argument("value", type=str, help="new value for config property")


def _build_prepare(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("protobuf_path", type=str, help="path to the SceneNet RGB-D protobuf file")
    parser.add_argument("ground_truth_path", type=str,
                        help="path to store ground truth - relative to configured ground truth path")
    

def _build_train(parser: argparse.ArgumentParser) -> None:
    sub_parsers = parser.add_subparsers(dest="network")
    sub_parsers.required = True
    
    ssd_parser = sub_parsers.add_parser("ssd", help="SSD")
    
    # build sub parsers
    _build_ssd_train(ssd_parser)
 

def _build_ssd_train(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("num_epochs", type=int, help="the number of epochs to train", default=80)
    parser.add_argument("iteration", type=int, help="the training iteration")
    

def _build_test(parser: argparse.ArgumentParser) -> None:
    sub_parsers = parser.add_subparsers(dest="network")
    sub_parsers.required = True
    
    ssd_bayesian_parser = sub_parsers.add_parser("bayesian_ssd", help="SSD with dropout layers")
    ssd_parser = sub_parsers.add_parser("ssd", help="SSD")
    
    # build sub parsers
    _build_ssd_test(ssd_bayesian_parser)
    _build_ssd_test(ssd_parser)


def _build_ssd_test(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("iteration", type=int, help="the validation iteration")
    parser.add_argument("train_iteration", type=int, help="the train iteration")


def _build_evaluate(parser: argparse.ArgumentParser) -> None:
    sub_parsers = parser.add_subparsers(dest="network")
    sub_parsers.required = True

    ssd_bayesian_parser = sub_parsers.add_parser("bayesian_ssd", help="SSD with dropout layers")
    ssd_parser = sub_parsers.add_parser("ssd", help="SSD")

    # build sub parsers
    _build_ssd_evaluate(ssd_bayesian_parser)
    _build_ssd_evaluate(ssd_parser)


def _build_ssd_evaluate(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("iteration", type=int, help="the validation iteration to use")


def _build_visualise(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("tarball_id", type=str, help="id of the used tarball. number for training tarball or 'test'")
    parser.add_argument("trajectory", type=int, help="trajectory to visualise")


def _build_visualise_metrics(parser: argparse.ArgumentParser) -> None:
    sub_parsers = parser.add_subparsers(dest="network")
    sub_parsers.required = True
    
    ssd_bayesian_parser = sub_parsers.add_parser("bayesian_ssd", help="SSD with dropout layers")
    ssd_parser = sub_parsers.add_parser("ssd", help="SSD")
    
    ssd_bayesian_parser.add_argument("iteration", type=int, help="the validation iteration to use")
    ssd_parser.add_argument("iteration", type=int, help="the validation iteration to use")


def _build_visualise_all(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("vanilla_ssd_001_iteration", type=int,
                        help="validation iteration of vanilla SSD with 0.01 conf threshold")
    parser.add_argument("vanilla_ssd_02_iteration", type=int,
                        help="validation iteration of vanilla SSD with 0.2 conf threshold")
    parser.add_argument("entropy_ssd_001_iteration", type=int,
                        help="validation iteration of vanilla SSD with 0.01 conf threshold and entropy threshold")
    parser.add_argument("bayesian_ssd_no_do_no_nms_iteration", type=int,
                        help="validation iteration of Bayesian SSD with disabled dropout and without NMS")
    parser.add_argument("bayesian_ssd_no_do_iteration", type=int,
                        help="validation iteration of Bayesian SSD with disabled dropout")
    parser.add_argument("bayesian_ssd_do_09_iteration", type=int,
                        help="validation iteration of Bayesian SSD with dropout keep rate 0.9")
    parser.add_argument("bayesian_ssd_do_05_iteration", type=int,
                        help="validation iteration of Bayesian SSD with dropout keep rate 0.5")


def _build_measure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("tarball_id", type=str, help="id of the used tarball. number for training tarball or 'test'")


if __name__ == "__main__":
    main()
