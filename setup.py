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

"""setup file for masterthesis"""
from setuptools import find_packages, setup

with open("README.md", "rb") as f:
    long_desc = f.read().decode()

setup(
    name="twomartens.masterthesis",
    description="Code of the master thesis",
    long_description=long_desc,
    long_description_content_type="text/markdown; charset=UTF-8",
    author="Jim Martens",
    author_email="github@2martens.de",
    url="https://git.2martens.de/2martens/masterthesis",
    version="0.1.0",
    namespace_packages=["twomartens"],
    packages=find_packages('src', exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    package_dir={'': 'src'},
    package_data={},
    entry_points={},
    python_requires="~=3.6",
    install_requires=["tensorflow"],
    license="Apache License 2.0",
    classifiers=[
        "Operating System :: OS Independent",
        "Development Status :: 1 - Planning",
        "License :: OSI Approved :: Apache Software License",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
)
