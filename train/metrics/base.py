import logging
import os
from pathlib import Path


def create_log(logpath=None):
    """
    Set up logging for terminal and log file. This is optional and typically not used when ML Tracking
    is saving terminal outputs, anyway.

    Parameters
    ----------
    logpath  :  str folder to save log. Will be auto created.

    """
    logger = logging.getLogger('neurodiag')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s: %(message)s')  # controls onscreen and file logs

    if logpath:
        Path(logpath).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(os.path.join(logpath, 'neurodiag.log'))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger
