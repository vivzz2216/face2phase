"""
Centralized logging configuration for Face2Phrase
"""
import logging
import sys
from pathlib import Path
from .settings import PROJECT_ROOT

# Configure logging
LOG_FILE = PROJECT_ROOT / "face2phrase.log"

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding='utf-8')
    ],
    force=True
)

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given name"""
    return logging.getLogger(name)

