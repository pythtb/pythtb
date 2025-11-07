__version__ = "2.0.0"
__author__ = "Trey Cole, Sinisa Coh, David Vanderbilt"
__license__ = "GPL-3.0"

# Set up logging
import logging

_LOGGER_NAME = __name__.split(".")[0]
logger = logging.getLogger(_LOGGER_NAME)
logger.addHandler(logging.NullHandler())
_DEFAULT_LOG_FORMAT = "%(levelname)s %(name)s: %(message)s"
_DEFAULT_HANDLER: logging.Handler | None = None


def _coerce_level(level: int | str) -> int:
    """Normalize a logging level provided as an int or name."""
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        try:
            return logging._nameToLevel[level.upper()]
        except KeyError as exc:  # pragma: no cover - defensive shim
            raise ValueError(f"Unknown logging level: {level}") from exc
    raise TypeError("Logging level must be an int or one of the named levels.")


def configure_logging(
    level: int | str = "INFO",
    *,
    handler: logging.Handler | None = None,
    fmt: str | None = _DEFAULT_LOG_FORMAT,
    propagate: bool = False,
) -> logging.Handler:
    """Configure logging for the pythtb package in a single call."""
    global _DEFAULT_HANDLER

    pkg_logger = logging.getLogger(_LOGGER_NAME)
    numeric_level = _coerce_level(level)

    # Drop NullHandlers; we manage our own default.
    pkg_logger.handlers = [
        h for h in pkg_logger.handlers if not isinstance(h, logging.NullHandler)
    ]

    # Reuse the existing handler (default or user-supplied) instead of stacking new ones.
    if handler is None:
        if _DEFAULT_HANDLER is None:
            _DEFAULT_HANDLER = logging.StreamHandler()
        handler = _DEFAULT_HANDLER

    for existing in list(pkg_logger.handlers):
        if existing is not handler:
            pkg_logger.removeHandler(existing)

    if handler not in pkg_logger.handlers:
        pkg_logger.addHandler(handler)

    if fmt:
        handler.setFormatter(logging.Formatter(fmt))

    pkg_logger.setLevel(numeric_level)
    pkg_logger.propagate = propagate

    return handler


def set_log_level(level: int | str):
    """Set logging level for all pythtb loggers.

    Installs a default stream handler the first time it is called so messages
    become visible immediately.
    """
    configure_logging(level=level)


# Import public symbols
from . import tbmodel, wfarray, w90, mesh, wannier, utils, lattice
from .tbmodel import *  # relies on tbmodel.__all__
from .wfarray import *
from .w90 import *
from .mesh import *
from .wannier import *
from .utils import *
from .lattice import *

# Use the core module's __all__ to define the package exports from * imports.

__all__ = []
for m in (tbmodel, wfarray, w90, mesh, wannier, utils, lattice):
    __all__ += getattr(m, "__all__", [])
