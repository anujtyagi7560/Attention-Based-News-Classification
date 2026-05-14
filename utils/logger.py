# utils/logger.py
"""
Centralised logging setup.
Call get_logger(__name__) in any module to get a named logger.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_initialized = False


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> None:
    """
    Configure root logger once.

    Args:
        level    : Logging level (e.g. logging.DEBUG).
        log_file : Optional path; if given, logs are written there too.
    """
    global _initialized
    if _initialized:
        return

    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a"))

    logging.basicConfig(
        level=level,
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
        handlers=handlers,
    )
    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger, initialising root logger if needed.

    Args:
        name: Typically __name__ of the calling module.
    """
    setup_logging()
    return logging.getLogger(name)
