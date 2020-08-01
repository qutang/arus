"""
Module that provides environment variable and path settings for arus package

Author: Qu Tang
Date: 01/28/2020
License: GNU v3
"""


import os

_ARUS_DATA_ = 'ARUS_DATA'
_CACHE_ = 'CACHE'


def get_data_home(data_home=None):
    """Return the path of the seaborn data directory.
    This is used by the ``load_dataset`` function.
    If the ``data_home`` argument is not specified, the default location
    is ``~/arus-data``.
    Alternatively, a different default location can be specified using the
    environment variable ``ARUS_DATA``.
    """
    if data_home is None:
        data_home = os.environ.get(_ARUS_DATA_,
                                   os.path.join('~', 'arus-data'))
    data_home = os.path.expanduser(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    return data_home


def get_cache_home(cache_home=None):
    if cache_home is None:
        cache_home = os.environ.get(_CACHE_,
                                    os.path.join('~', 'arus-cache'))
    cache_home = os.path.expanduser(cache_home)
    if not os.path.exists(cache_home):
        os.makedirs(cache_home)
    return cache_home
