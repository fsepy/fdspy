# fdspy

Python tools for [Fire Dynamics Simulator](https://github.com/firemodels/fds).

## Getting Started

Python 3.6 or later is required.

### Installation

Chose one of the following installation path.

[Python](https://www.python.org/downloads/) 3.7 or later is required. [Anaconda Distribution](https://www.anaconda.com/distribution/#download-section) is recommended for new starters, it includes Python and few useful packages including a package management tool pip (see below).

[pip](https://pypi.org/) is a package management system for installing and updating Python packages. pip comes with Python, so you get pip simply by installing Python. On Ubuntu and Fedora Linux, you can simply use your system package manager to install the `python3-pip` package. [The Hitchhiker's Guide to Python](https://docs.python-guide.org/starting/installation/) provides some guidance on how to install Python on your system if it isn't already; you can also install Python directly from [python.org](https://www.python.org/getit/). You might want to [upgrade pip](https://pip.pypa.io/en/stable/installing/) before using it to install other programs.

1. to use `pip` install from PyPI:

    [![Downloads](https://pepy.tech/badge/fdspy)](https://pepy.tech/project/fdspy)

    ```sh
    pip install --upgrade fdspy
    ```

2. to use `pip` install from GitHub (requires [git](https://git-scm.com/downloads)):  

    *Note installing `fdspy` via this route will include the latest commits/changes to the library.*  

    ```sh
    pip install --upgrade "git+https://github.com/fsepy/SfePrapy.git@master"
    ```

#### Local
For any reasons you do not want to install `fdspy` as a Python library, download the entire project folder to your computer (e.g. Desktop), change current working directory to the project folder then install `fdspy` by using pip:

```bash
pip install .
```

The installation process should be able to install all required libraries listed in Prerequisites.

### Usage

todo

#### FDS Job Scheduler

todo

#### Generate Devices in Defined Domain

todo

#### MTR Calculation

todo

## Authors

* **Yan Fu** - *fuyans@gmail.com*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
