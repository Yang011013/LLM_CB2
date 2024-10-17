import logging
logger = logging.getLogger('project_logger')
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('logs/game.log')
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s [%(module)s:%(funcName)s:%(lineno)d] %(message)s")
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)