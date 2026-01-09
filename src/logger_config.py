import logging
import os
import sys
from datetime import datetime

from termcolor import colored


class ColoredFormatter(logging.Formatter):
    """Custom formatter for colored console output."""

    def format(self, record):
        msg = super().format(record)
        if record.levelno == logging.WARNING:
            return colored(msg, "yellow")
        elif record.levelno == logging.ERROR:
            return colored(msg, "red")
        elif record.levelno == logging.CRITICAL:
            return colored(msg, "red", attrs=["bold"])
        elif record.levelno == logging.INFO:
            # Keep info mostly neutral/white unless specific logic applies,
            # or rely on message content coloring (which we do manually in calls).
            return msg
        return msg


def setup_logger(log_dir="logs"):
    """Sets up a logger that writes to a timestamped file and stdout."""

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"rlm_run_{timestamp}.log")

    # Root logger configuration
    logger = logging.getLogger("RLM")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers to prevent duplicate logs

    # File Handler (Detailed, Plain Text)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)  # Capture everything in file
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console Handler (Summary, Colored)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = ColoredFormatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.info(f"Logging initialized. Log file: {log_file}")

    return logger, log_file
