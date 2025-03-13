import logging

def setup_logger(log_file, level):
    if level == "info":
        loglev = logging.INFO
    elif level == "warning":
        loglev = logging.WARN
    elif level == "debug":
        loglev = logging.DEBUG
    logger = logging.getLogger()
    logger.setLevel(loglev)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(loglev)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(loglev)

    formatter = logging.Formatter("%(asctime)s  - %(levelname)s - %(message)s")

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
