import logging
import sys


def setup_logger(name: str = "app", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger
