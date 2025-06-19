import os
from pathlib import Path
from itertools import islice

import yaml


def take(n, iterable):
    """Return the first n items of the iterable as a list."""
    return list(islice(iterable, n))


def get_project_root():
    """Get absolute path of the project"""
    return str(Path(__file__).resolve().parent.parent)


def get_package_root():
    """Get absolute path of the package"""
    return str(Path(__file__).resolve().parent)


def get_config():
    """Read config file"""
    with open(os.path.join(get_package_root(), "config.yml")) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config
