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
        "weights": (str, "")
    },
    "Debug": {
        "summaries": (bool, "True"),
        "train_images": (bool, "False")
    },
    "Parameters": {
        "batch_size": (int, "32"),
        "ssd_image_size": (int, "300")
    }
}


def get_property(key: str) -> str:
    parser = configparser.ConfigParser()
    config_file = f"{os.getcwd()}/{CONFIG_FILE}"
    
    _initialise_config(config_file)
    parser.read(config_file)
    
    section, prop = tuple(key.split("."))
    return parser.get(section, prop)


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
            pass
    except FileNotFoundError:
        with open(config_file, "w") as file:
            parser = configparser.ConfigParser()
            for section in _CONFIG_PROPS:
                parser[section] = {}
                for option in _CONFIG_PROPS[section]:
                    _, default = _CONFIG_PROPS[section][option]
                    parser[section][option] = default
                    
            parser.write(file)
