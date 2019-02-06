#!/usr/bin/env python

# from distutils.core import setup

import os
import setuptools

# To use a consistent encoding
from codecs import open

# Get the long description from the README file
with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'README.md')) as file:
    long_description = file.read()

setuptools.setup(
    name='fdspy',

    version='0.0.8',

    description='Fire Dynamics Simulator Python',

    author='Yan Fu',

    author_email='fuyans@gmail.com',

    url='https://github.com/fsepy/fdspy',

    download_url="https://github.com/fsepy/fdspy/archive/master.zip",

    keywords=["fire safety", "structural fire engineering", "fire dynamics simulator", "computational fluid dynamics", "fds", "cfd"],

    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 3 - Alpha",
        "Environment :: Other Environment",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],

    long_description='Just a wrapper with additional functions to automate things.',

    packages=[
        'fdspy',
        'fdspy.calc_mtr',
        'fdspy.calc_sprinkler',
        'fdspy.fdsback',
        'fdspy.fdsfront',
        'fdspy.gen_devc',
        'fdspy.fdsbginfo'
    ],

    # entry_points={'console_scripts': ['sfeprapymc = sfeprapy.mc.__main__:run']},

    install_requires=[
        'numpy>=1.15.0',
        'pandas>=0.23.3',
        'scipy>=1.1.0',
        'seaborn>=0.9.0',
        'pyside2>=5.12'
    ],

    include_package_data=True,
)
