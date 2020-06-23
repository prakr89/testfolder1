from importlib import reload
import logging
reload(logging)
logging.basicConfig(filename=r"C:\Users\M1053110\github\mycodings\pck\test.log",level=logging.DEBUG,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d')
logger=logging.getLogger(__name__)
def name():
    print("Praveen")
    logger.debug("1")
