import logging
import sys

DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s -   %(message)s"
DEFAULT_LOG_DATE = "%m/%d/%Y %H:%M:%S"
DEFAULT_LOG_FORMATTER = logging.Formatter(DEFAULT_LOG_FORMAT, DEFAULT_LOG_DATE)

logging.basicConfig(
    format=DEFAULT_LOG_FORMAT,
    datefmt=DEFAULT_LOG_DATE,
    handlers=[logging.StreamHandler(sys.stdout)],
)


def get_logger(name: str, level=DEFAULT_LOG_LEVEL) -> logging.Logger:
    """Initialize and return a named logger with the specified log level.

    Args:
        name: Name for the logger instance
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    named_logger = logging.getLogger(name=name)
    named_logger.setLevel(level)

    # named_logger.addHandler(LOG_HANDLER)
    return named_logger
