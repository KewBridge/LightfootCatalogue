import logging
import sys


def get_logger(name=__name__, level=logging.INFO):
    """
    Returns a logger with the specified name and logging level.
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



