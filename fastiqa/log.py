# import logging
# # logging.basicConfig(level=logging.WARNING)
# logging.basicConfig(level=logging.DEBUG)


# import sys
# from loguru import logger
# logger.remove()
# logger.add(sys.stderr, level="DEBUG")
# logger.debug('haha')
# logger.info('haha')


import sys
from loguru import logger

DEBUG = False

if DEBUG:
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
else:
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
