from importlib import reload
import logging
reload(logging)
##Importing module mtd1
# import os.path
# import sys
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
#sys.path.append(os.path.dirname(os.path.abspath(__file__)))
###Importing modile(filename) mtd2
import sys
print(sys.path)
import os
print(os.getcwd())
#C:\Users\M1053110\github\mycodings

#for linux
sys.path.insert(0, "./pythonws")
sys.path.remove("./pythonws")
sys.path.insert(0, r"C:\Users\M1053110\pythonws")
sys.path.remove(r"C:\Users\M1053110\pythonws")

##Importing module using .(dot) notation
try:
    from pck.subpck.prav import *
    from pck.subpck.bahubali import *
except ImportError:
    print('No Import')

def main():
    pizza_01 = Pizza("Sicilian", 18)
    pizza_01.make(5)
    pizza_01.eat(4)
    name()
    pizza_02 = Pizza("quattro formaggi", 16)
    pizza_02.make(2)
    pizza_02.eat(2)
    logger.debug("2")
#if __name__ == '__main__' and __package__ is None:
if __name__ == '__main__':
    logging.basicConfig(filename=r"C:\Users\M1053110\github\mycodings\pck\test.log",level=logging.DEBUG,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d')
    logger=logging.getLogger(__name__)
    main()
##run using terminal as ===python -m mypackage.main
## Linux & OSX
#export PYTHONPATH=$HOME/dirWithScripts/:$PYTHONPATH
## Windows
#set PYTHONPATH=C:\path\to\dirWithScripts\;%PYTHONPATH%
"""
class Rectangle:
   def __init__(self, length, breadth, unit_cost=0):
       self.length = length
       self.breadth = breadth
       self.unit_cost = unit_cost

   def get_perimeter(self):
       return 2 * (self.length + self.breadth)

   def get_area(self):
       return self.length * self.breadth

   def calculate_cost(self):
       area = self.get_area()
       return area * self.unit_cost
# breadth = 120 cm, length = 160 cm, 1 cm^2 = Rs 2000
r = Rectangle(160, 120, 2000)
print("Area of Rectangle: %s cm^2" % (r.get_area()))
print("Cost of rectangular field: Rs. %s " %(r.calculate_cost()))

"""
