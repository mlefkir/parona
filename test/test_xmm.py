"""unit tests for core.py module
"""

import unittest

import sys
sys.path.append('../../src')



class Test_XMM_Newton(unittest.TestCase):
    
    def setUp(self):
        self.obsid = "0142830101"
        self.path = "data_test"
