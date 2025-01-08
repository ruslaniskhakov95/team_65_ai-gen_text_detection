import logging
from logging.handlers import RotatingFileHandler
from typing import Any, Dict


def logger_setup(name: str, log_file: str, level: Any = logging.INFO,
                 logger_args: Dict[str, Any] = {
                    "maxBytes": 10**6, "backupCount": 5
                 }) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    handler = RotatingFileHandler(log_file, **logger_args)
    handler.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


logger = logger_setup(
    name='streamlit_logger', log_file='streamlit_app/logs_streamlit/log.log'
)
