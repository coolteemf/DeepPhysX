import unittest
from os import devnull
from sys import stdout

from tests_Core import *
from tests_Torch import *
from tests_Sofa import *

if __name__ == '__main__':
    stdout = open(devnull, 'w')
    unittest.main()
