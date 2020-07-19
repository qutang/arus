"""
__arus__ python package provides a computation framework to manage and process ubiquitous sensory data for activity recognition.

"""

__version__ = '1.1.11'

from . import accelerometer as accel
from . import cli, dataset
from . import developer as dev
from . import env
from . import extensions as ext
from . import mhealth_format as mh
from . import (operator, plugins, processor, segmentor, stream2, synchronizer,
               testing)
from .generator import *
from .moment import Moment
from .node import Node
from .scheduler import Scheduler
from .stream import Stream
from . import models
