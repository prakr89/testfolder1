#If you set the log level to INFO, it will not include NOTSET, DEBUG messages.
# Level	     Numeric value
# CRITICAL	  50
# ERROR	      40
# WARNINGG	  30---This is default
# INFO	      20
# DEBUGG	  10
# NOTSET   	  0

#Capturing Stack Tracesback:- fro this use:- exc_info=True

#Method 2 Using importlib reload module to activate logging as it is depriciated in python 3
from importlib import reload
import logging
reload(logging)
logging.basicConfig(filename=r"C:\Users\M1053110\mycodings\test.log", mode='a',level=logging.INFO,format='%(asctime)s - %(message)s')
name = 'John'
logging.debug('This is a debug1 message')
logging.info('This is an info1 message')
logging.warning('This is a warning1 message')
logging.error(f'{name}--This is an error1 message')
logging.critical('This is a critical1 message')


#Method 2 Using setLevel
import logging
logger = logging.getLogger()
fhandler = logging.FileHandler(filename=r"C:\Users\M1053110\mycodings\test.log")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fhandler.setFormatter(formatter)
logger.addHandler(fhandler)
logger.setLevel(logging.DEBUG)
name = 'John'
logging.debug('This is a debug2 message')
logging.info('This is an info2 message')
logging.warning('This is a warning2 message')
logging.error(f'{name}--This is an error2 message')
logging.critical('This is a critical2 message')
logger.removeHandler(fhandler)
logger.removeHandler(formatter)
#root.removeHandler(handler)
#logger.propagate = False

#Method 3 using sys.stdout
import logging
import sys
filepath =r"C:\Users\M1053110\mycodings\test.log"
filename = logging.FileHandler(filepath,"w+")
stream = logging.StreamHandler(sys.stdout)
handlers = [file_handler, stream_handler]
#below mode ="a" is by default but note that it will not overwrite even mode ="w" because it require python shell restart
#to ovwerwrite file in same session use w+ in FileHandler
logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s',level=logging.INFO, handlers=handlers)
name = 'John'
logging.debug('This is a debug3 message')
logging.info('This is an info3 message')
logging.warning('This is a warning3 message')
logging.error(f'{name}--This is an error3 message')
logging.critical('This is a critical3 message')


#Demo basic
import logging

a = 5
b = 0

try:
  c = a / b
except Exception as e:
  logging.error("Exception occurred", exc_info=T)
