import logging
import os
from pathlib import Path


def define_logger(log_dir, log_name):
    formatter = logging.Formatter('%(message)s')
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    # console_handler = logging.StreamHandler()
    # console_handler.formatter = formatter
    # console_handler.setLevel(logging.INFO)
    # logger.addHandler(console_handler)

    file_dir = Path(log_dir)
    file_dir.mkdir(exist_ok=True, parents=True)
    file_path = os.path.join(log_dir, log_name)
    file_handler = logging.FileHandler(file_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    return logger
