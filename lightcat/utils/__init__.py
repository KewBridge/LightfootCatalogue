import logging
import sys

def get_logger(name: str = __name__, level: int = logging.INFO):
    """
    Returns a logger with the specified name and logging level.

    Args:
        name (str, optional): file name. Defaults to __name__.
        level (int, optional): logging level. Defaults to logging.INFO.

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not(logger.hasHandlers()):

        # Create a console handler to output logs to the console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.propagate = False  # Prevents the logger from propagating to the root logger
    

    return logger



