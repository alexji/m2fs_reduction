#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from setuptools import setup, find_packages
from codecs import open
from os import path, system
from re import compile as re_compile

# For convenience.
if sys.argv[-1] == "publish":
    system("python setup.py sdist upload")
    sys.exit()

def read(filename):
    kwds = {"encoding": "utf-8"} if sys.version_info[0] >= 3 else {}
    with open(filename, **kwds) as fp:
        contents = fp.read()
    return contents

# Get the version information.
here = path.abspath(path.dirname(__file__))
vre = re_compile("__version__ = \"(.*?)\"")
version = vre.findall(read(path.join(here, "m2fs_reduction", "__init__.py")))[0]

setup(
    name="m2fs_reduction",
    version=version,
    author="Alex Ji",
    description="M2FS Reduction Pipeline",
    long_description=read(path.join(here, "README.md")),
    url="https://github.com/alexji/m2fs_reduction",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
    ],
    keywords="astronomy",
    packages=find_packages(exclude=["documents", "tests"]),
    install_requires=[
        "numpy","scipy","matplotlib","astropy"
        ],
)
