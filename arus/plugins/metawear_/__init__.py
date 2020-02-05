try:
    from .stream import *
    from .scanner import *
    from .corrector import *
except ImportError as e:
    msg = (
        "Arus plugin metawear requirements are not installed.\n\n"
        "Please install the metawear extra packages as follows:\n\n"
        "  pip install arus[metawear]\n\n"
        "  poetry add arus --extras metawear"
    )
    raise ImportError(str(e) + "\n\n" + msg)