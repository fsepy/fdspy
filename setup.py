#!/usr/bin/env python

# from distutils.core import setup

import os
import setuptools
import fdspy

from codecs import open  # To use a consistent encoding


with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "README.md")) as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="fdspy",
    version=fdspy.__version__,
    description="Fire Dynamics Simulator Python",
    author="Yan Fu",
    author_email="fuyans@gmail.com",
    url="https://github.com/fsepy/fdspy",
    download_url="https://github.com/fsepy/fdspy/archive/master.zip",
    keywords=[
        "fire",
        "engineering" "fire dynamics simulator",
        "computational fluid dynamics",
        "fds",
        "cfd",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 3 - Alpha",
        "Environment :: Other Environment",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],
    long_description=long_description,
    packages=[
        "fdspy",
        "fdspy.calc_mtr",
        "fdspy.calc_sprinkler",
        "fdspy.fdsback",
        "fdspy.fdsfront",
        "fdspy.gen_devc",
        "fdspy.fdsbginfo",
        "fdspy.lib",
    ],
    install_requires=requirements,
    include_package_data=True,
    entry_points={"console_scripts": ["fdspy=fdspy.cli:main"]},
)
