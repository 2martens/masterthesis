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

Functions:
    get_property(key: str): returns the value of given property
    set_property(key: str, value: str): sets the given property to given value
    list_property_values(): prints out list of config values
    get_config(): returns key-value store
"""
from typing import Union, Dict


def get_property(key: str) -> str:
    raise NotImplementedError


def set_property(key: str, value: str) -> None:
    raise NotImplementedError


def list_property_values() -> None:
    raise NotImplementedError


def get_config() -> Dict[str, Union[str, int, float, bool]]:
    raise NotImplementedError
