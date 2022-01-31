#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import glob
import os
import shutil
from setuptools import setup
import torch

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 7], "Requires PyTorch >= 1.7"


setup(
    name="maskformer",
    author="",
    description="MaskFormer"
    "platform for object detection and segmentation.",
    python_requires=">=3.6",
    install_requires=[
        "Pillow>=7.1",
        "matplotlib",
        "pycocotools>=2.0.2",
        "termcolor>=1.1",
        "yacs>=0.1.8",
        "tabulate",
        "cloudpickle",
        "tqdm>4.29.0",
        "tensorboard",
        "fvcore>=0.1.5,<0.1.6",
        "iopath>=0.1.7,<0.1.10",
        "future",
        "black==21.4b2",
        "scipy>1.5.1",
    ],
)