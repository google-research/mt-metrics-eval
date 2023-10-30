# coding=utf-8
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Setup script for mt-metrics-eval.

This script will allow pip-installing as a Python module.
"""

import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

install_requires = [
    "apache_beam",
    "numpy",
    "scipy",
    "absl-py",
]

setuptools.setup(
    name="mt-metrics-eval",
    version="0.0.3",
    author="George Foster",
    description="Toolkit for evaluating Machine Translation metrics.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/google-research/mt-metrics-eval",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
    ],
    license="Apache 2.0",
    python_requires=">=3",
    install_requires=install_requires)
