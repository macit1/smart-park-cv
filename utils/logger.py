import logging
import sys


def setup_logger(name="ParkSense", debug_mode=False):
    logger = logging.getLogger(name)

    if logger.hasHandlers():
        logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)
        return logger

    logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if debug_mode else logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.propagate = False

    return logger


logger = logging.getLogger("ParkSense")
