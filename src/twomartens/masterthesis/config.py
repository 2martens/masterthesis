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
Takes care of config functionality.

Constants:
    CONFIG_FILE: name of config file relative to working directory

Functions:
    get_property(key: str): returns the value of given property (e.g. "section.option")
    set_property(key: str, value: str): sets the given property to given value
    list_property_values(): prints out list of config values
    get_config(): returns key-value store
"""
import configparser
import os
from typing import Union

from attributedict.collections import AttributeDict

CONFIG_FILE = "tm-masterthesis-config.ini"
_CONFIG_PROPS = {
    "Paths": {
        "coco": (str, ""),
        "scenenet": (str, ""),
        "scenenet_gt": (str, ""),
        "scenenet_gt_train": (str, ""),
        "scenenet_gt_val": (str, ""),
        "scenenet_gt_test": (str, ""),
        "output": (str, ""),
        "summaries": (str, ""),
        "weights": (str, ""),
        "evaluation": (str, "")
    },
    "Debug": {
        "summaries": (bool, "True"),
        "train_images": (bool, "False")
    },
    "Parameters": {
        "batch_size": (int, "32"),
        "ssd_image_size": (int, "300"),
        "learning_rate": (float, "0.001"),
        "steps_per_val_epoch": (int, "20"),
        "ssd_forward_passes_per_image": (int, "10"),
        "nr_classes": (int, "80"),
        "ssd_confidence_threshold": (float, "0.5"),
        "ssd_iou_threshold": (float, "0.45"),
        "ssd_top_k": (int, "200"),
        "ssd_dropout_rate": (float, "0.5"),
        "ssd_use_nms": (bool, "True"),
        "ssd_use_entropy_threshold": (bool, "False"),
        "ssd_entropy_threshold_min": (float, "0.1"),
        "ssd_entropy_threshold_max": (float, "2.5"),
        "ssd_test_pretrained": (bool, "False"),
        "ssd_use_coco": (bool, "False"),
        "ssd_use_dropout": (bool, "True"),
        "nr_trajectories": (int, "-1")
    }
}


class Config:
    """
    Data class for config values.
    """
    def __init__(self):
        self.paths = AttributeDict({})
        self.debug = AttributeDict({})
        self.parameters = AttributeDict({})
        for group in _CONFIG_PROPS:
            for value in _CONFIG_PROPS[group]:
                self[group.lower()][value] = get_property(f"{group}.{value}")
    
    def __getitem__(self, item):
        return getattr(self, item)


def get_property(key: str) -> Union[str, float, int, bool]:
    parser = configparser.ConfigParser()
    config_file = f"{os.getcwd()}/{CONFIG_FILE}"
    
    _initialise_config(config_file)
    parser.read(config_file)
    
    section, prop = tuple(key.split("."))
    cast = _CONFIG_PROPS[section][prop][0]
    
    value = None
    if cast is str:
        value = parser.get(section, prop)
    elif cast is float:
        value = parser.getfloat(section, prop)
    elif cast is int:
        value = parser.getint(section, prop)
    elif cast is bool:
        value = parser.getboolean(section, prop)
    
    return value


def set_property(key: str, value: str) -> None:
    parser = configparser.ConfigParser()
    config_file = f"{os.getcwd()}/{CONFIG_FILE}"
    
    _initialise_config(config_file)
    parser.read(config_file)
    
    section, prop = tuple(key.split("."))
    parser.set(section, prop, value)
    
    with open(config_file, "w") as file:
        parser.write(file)


def list_property_values() -> None:
    parser = configparser.ConfigParser()
    config_file = f"{os.getcwd()}/{CONFIG_FILE}"
    
    _initialise_config(config_file)
    parser.read(config_file)
    
    for section in parser:
        print(f"[{section}]")
        for option in parser[section]:
            value = parser.get(section, option)
            print(f"{option}: {value}")


def get_config() -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    config_file = f"{os.getcwd()}/{CONFIG_FILE}"
    _initialise_config(config_file)
    parser.read(config_file)
    
    return parser


def _initialise_config(config_file: str) -> None:
    # work-around for implementation detail of config parser
    # a non-existing file does not lead to an exception but is simply ignored
    # therefore a manual check via this construction is required
    
    try:
        with open(config_file, "r"):
            # if we reach this branch then the file exists and everything is fine
            return
    except FileNotFoundError:
        with open(config_file, "w") as file:
            parser = configparser.ConfigParser()
            for section in _CONFIG_PROPS:
                parser[section] = {}
                for option in _CONFIG_PROPS[section]:
                    _, default = _CONFIG_PROPS[section][option]
                    parser[section][option] = default
                    
            parser.write(file)
