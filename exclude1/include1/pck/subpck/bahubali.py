from importlib import reload
import logging
reload(logging)
logging.basicConfig(filename=r"C:\Users\M1053110\github\mycodings\pck\test.log",level=logging.DEBUG,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d')
logger=logging.getLogger(__name__)
class Pizza():
    def __init__(self, name, price):
        self.name = name
        self.price = price
        logger.debug("Pizza created: {} (${})".format(self.name, self.price))

    def make(self, quantity=1):
        logger.debug("Made {} {} pizza(s)".format(quantity, self.name))

    def eat(self, quantity=1):
        logger.debug("Ate {} pizza(s)".format(quantity, self.name))
