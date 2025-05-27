"""
Centralized Loguru logger setup.
"""
import os
import sys
from pathlib import Path
from loguru import logger

def setup_logging(save_dir: str, level: str = "INFO"):
    """
    Configure Loguru to log to stderr and to a rolling file in save_dir.

    Args:
        save_dir: Directory where log file will be written.
        level: Logging level (e.g., "INFO", "DEBUG").
    """
    # remove default handler
    logger.remove()
    # console logger
    logger.add(sys.stderr, level=level, format="<level>{message}</level>")
    # ensure save_dir exists
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    # file logger with time and level
    log_file = Path(save_dir) / "train.log"
    logger.add(
        str(log_file),
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        rotation="10 MB",
        retention="7 days",
    )
    logger.info(f"Logging initialized. Level={level}, dir={save_dir}")
