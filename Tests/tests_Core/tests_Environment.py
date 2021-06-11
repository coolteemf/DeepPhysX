import unittest
import numpy as np

from DeepPhysX.Environment.BaseEnvironment import BaseEnvironment
from DeepPhysX.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX.Manager.EnvironmentManager import EnvironmentManager
from Tests.tests_Core.numpy_environment import NumpyEnvironment


class TestBaseEnvironmentConfig(unittest.TestCase):

    def setUp(self):
        pass

    def test_init(self):
        with self.assertRaises(TypeError):
            BaseEnvironmentConfig(simulations_per_step=0)
            BaseEnvironmentConfig(simulations_per_step='one')
            BaseEnvironmentConfig(max_wrong_samples_per_step=-1)
            BaseEnvironmentConfig(max_wrong_samples_per_step='ten')
            BaseEnvironmentConfig(always_create_data='True')
            BaseEnvironmentConfig(multiprocessing=-1)
            BaseEnvironmentConfig(multiprocessing='two')
            BaseEnvironmentConfig(multiprocess_method='docker')

    def test_createDataset(self):
        # Bad
        config = BaseEnvironmentConfig(environment_class=BaseEnvironmentConfig)
        self.assertRaises(TypeError, config.createEnvironment)
        # Good
        BaseEnvironment.create = lambda obj, param: 0
        config = BaseEnvironmentConfig(environment_class=BaseEnvironment)
        self.assertIsInstance(config.createEnvironment(), BaseEnvironment)


class TestBaseEnvironment(unittest.TestCase):

    def setUp(self):
        self.config = BaseEnvironmentConfig().environment_config

    def test_notImplemented(self):
        with self.assertRaises(NotImplementedError):
            BaseEnvironment(self.config)
        BaseEnvironment.create = lambda obj, config: 0
        env = BaseEnvironment(self.config)
        with self.assertRaises(NotImplementedError):
            env.step()
            env.computeInput()
            env.computeOutput()


class TestEnvironmentManager(unittest.TestCase):

    def setUp(self):
        self.environment_config = BaseEnvironmentConfig(environment_class=NumpyEnvironment)
