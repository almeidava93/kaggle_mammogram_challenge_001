import logging
import os
from logging.handlers import RotatingFileHandler

def get_logger(name: str, log_level=logging.INFO, log_to_file=False, log_dir='logs', max_bytes=5_000_000, backup_count=3):
    """
    Returns a configured logger.

    Parameters:
        name (str): The name of the logger, typically `__name__`.
        log_level (int): Logging level (e.g., logging.DEBUG, logging.INFO).
        log_to_file (bool): Whether to log to a file in addition to the console.
        log_dir (str): Directory to store log files if log_to_file is True.
        max_bytes (int): Max size in bytes before rotating log file.
        backup_count (int): Number of backup files to keep.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if logger.hasHandlers():
        return logger  # Prevent adding multiple handlers

    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_to_file:
        os.makedirs(log_dir, exist_ok=True)
        file_path = os.path.join(log_dir, f"{name}.log")
        file_handler = RotatingFileHandler(file_path, maxBytes=max_bytes, backupCount=backup_count)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
