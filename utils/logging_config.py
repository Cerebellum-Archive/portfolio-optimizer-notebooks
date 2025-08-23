# logging_config.py
import logging
import os
import sys

def setup_logger(name: str = __name__) -> logging.Logger:
    """
    Creates and configures a logger.

    - Uses environment variable LOGLEVEL to set verbosity (default: INFO)
    - Works in scripts, Jupyter, and Emacs without duplication.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        # Avoid adding duplicate handlers if setup_logger is called multiple times
        return logger

    # Read log level from environment variable (default: INFO)
    log_level = os.getenv("LOGLEVEL", "INFO").upper()

    logger.setLevel(log_level)

    # Create handler that writes to stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)

    # Set a clean format: timestamp, level, message
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger
