__version__ = "2.0.0"
__author__ = "Trey Cole, Sinisa Coh, David Vanderbilt"
__license__ = "GPL-3.0"

# Set up logging
import logging

logger = logging.getLogger("pythtb")  # Main PythTB logger
logger.addHandler(logging.NullHandler())  # Avoids spurious warnings
logger.setLevel(logging.WARNING)  # Default level for library use


def get_logger():
    """Return the shared PythTB logger."""
    return logger


def enable_logging(level=logging.INFO):
    """Enable console logging at specified level (default: INFO)."""
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(ch)
    logger.setLevel(level)


def disable_logging():
    """Disable all logging output from PythTB."""
    logger.setLevel(logging.CRITICAL + 1)  # Effectively disables all logging


def set_logging_level(level):
    """Set logging level, e.g., logging.DEBUG, logging.INFO."""
    logger.setLevel(level)


# Import all public API from the core module
from .tb_model import *
from .wf_array import *
from .w90 import *
from .wannier import *
from .mesh import *
from .utils import *

from . import mesh, tb_model, wf_array, w90, wannier, utils

__all__ = []
__all__ += tb_model.__all__
__all__ += wf_array.__all__
__all__ += w90.__all__
__all__ += mesh.__all__
__all__ += wannier.__all__
__all__ += utils.__all__

# Use the core module's __all__ to define the package exports from * imports.
# This ensures 'from pythtb import *' pulls in only the intended public API.
# If you want to control what gets imported with "from pythtb import *",
# you can define __all__ in the respective modules (tb_model, wf_array, w90).
# This is a common practice in Python packages to avoid polluting the namespace
# with internal details and to provide a clear public API.
