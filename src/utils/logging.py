"""Logging utilities."""
import logging
from typing import Optional

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        level: Logging level.
        
    Returns:
        Configured logger.
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

