# logger.py
import logging
import os
from config import LOG_LEVEL, LOG_FILE, FLASK_DEBUG

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    # Set level from config
    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Always log to terminal
    console_handler = logging.StreamHandler()
    # In debug mode show everything; in production show WARNING and above
    console_handler.setLevel(logging.DEBUG if FLASK_DEBUG else logging.WARNING)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Always write INFO and above to log file
    if LOG_FILE:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)  # creates logs/ if missing
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger