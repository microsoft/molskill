import logging
import sys


# Adapted from https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/utils/logging.py
def get_logger(name: str = None, level: int = logging.INFO) -> logging.Logger:
    """Returns a logger that is configured as:
    - by default INFO level or higher messages are logged out in STDOUT.
    - format includes file name, line number, etc.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.hasHandlers():
        # Remove existing handlers so that capsys can capture
        # the output from patched sys.stdout
        for handler in logger.handlers:
            logger.removeHandler(handler)

    log_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s] %(message)s"
    )

    # Send everything to stdout
    handler_out = logging.StreamHandler(sys.stdout)
    handler_out.setFormatter(log_formatter)
    logger.addHandler(handler_out)

    return logger
