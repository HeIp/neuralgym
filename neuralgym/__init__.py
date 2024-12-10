import datetime

from logging.config import dictConfig as dictConfig
from .utils.logger import colorize as colorize

from . import callbacks as callbacks
from . import ops as ops
from . import train as train
from . import models as models
from . import data as data
from . import server as server

from .utils.gpus import set_gpus as set_gpus, get_gpus as get_gpus
from .utils.tf_utils import get_sess as get_sess
from .utils.config import Config as Config


__version__ = "0.0.1"
__all__ = ["Config", "get_gpus", "set_gpus", "date_uid", "unset_logger", "get_sess"]


def date_uid():
    """Generate a unique id based on date.

    Returns:
        str: Return uid string, e.g. '20171122171307111552'.

    """
    return (
        str(datetime.datetime.now())
        .replace("-", "")
        .replace(" ", "")
        .replace(":", "")
        .replace(".", "")
    )
