import numpy as np
import unittest
from core import debug
import time 


class Container:
    def __init__(self,i, x):
        self.x = x[i,:]
    
    def __repr__(self):
        return repr(self.x)

class TestingPythonDelete(unittest.TestCase):

    def test_delete(self):
        x = np.random.rand(5,4)
        debug("[TestingPythonDelete.test_delete()]  x= np.{}".format(repr(x)))
        c = Container(3,x)
        debug("[TestingPythonDelete.test_delete()] c = np.{}".format(repr(c)))
        del c 
        time.sleep(1) 
        debug("[TestingPythonDelete.test_delete()] x = np.{}".format(repr(x)))
        
