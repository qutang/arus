from ._model import *
from .muss2 import *
import pkg_resources


def get_prebuilt(model_name):
    return pkg_resources.resource_filename('arus', f'models/prebuilt/{model_name}.har')
