import unittest
import numpy as np

from DeepPhysX_Core.Environment.BaseEnvironment import BaseEnvironment
from DeepPhysX_Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from Tests.tests_Core.NumpyEnvironment import NumpyEnvironmentConfig


class TestBaseEnvironmentConfig(unittest.TestCase):

    def setUp(self):
        pass

    def test_init(self):
        # TypeError
        with self.assertRaises(TypeError):
            BaseEnvironmentConfig(simulations_per_step=1.5)
            BaseEnvironmentConfig(max_wrong_samples_per_step=9.5)
            BaseEnvironmentConfig(always_create_data='True')
        # ValueError
        with self.assertRaises(ValueError):
            BaseEnvironmentConfig(simulations_per_step=0)
            BaseEnvironmentConfig(max_wrong_samples_per_step=0)

    def test_createDataset(self):
        # ValueError
        class Test1:
            def __init__(self):
                pass
        environment_config = BaseEnvironmentConfig(environment_class=Test1)
        self.assertRaises(ValueError, environment_config.createEnvironment)
        # TypeError
        class Test2:
            def __init__(self, config):
                pass
        environment_config = BaseEnvironmentConfig(environment_class=Test2)
        self.assertRaises(TypeError, environment_config.createEnvironment)
        # No error
        class Test3(BaseEnvironment):
            def __init__(self, config):
                BaseEnvironment.__init__(self, config)
            def create(self):
                pass
        environment_config = BaseEnvironmentConfig(environment_class=Test3)
        self.assertIsInstance(environment_config.createEnvironment(), BaseEnvironment)


class TestBaseEnvironment(unittest.TestCase):

    def setUp(self):
        environment_config = BaseEnvironmentConfig().environment_config
        self.environment = BaseEnvironment(config=environment_config)

    def test_notImplemented(self):
        with self.assertRaises(NotImplementedError):
            self.environment.create()
            self.environment.step()
            self.environment.computeInput()
            self.environment.computeOutput()
        self.assertTrue(type(self.environment.getInput()) == np.ndarray)
        self.assertTrue(type(self.environment.getOutput()) == np.ndarray)


class TestCustomEnvironment(unittest.TestCase):

    def setUp(self):
        environment_config = NumpyEnvironmentConfig()
        self.environment = environment_config.createEnvironment()

    def test_create(self):
        self.assertTrue(self.environment.input_size == self.environment.output_size == 1)

    def test_step_reset(self):
        # Compute 10 steps
        for _ in range(10):
            self.environment.step()
        self.assertTrue(self.environment.idx_step == 10)
        # Reset
        self.environment.reset()
        self.assertTrue(self.environment.idx_step == 0)
        # Compute 10 * 2 steps
        self.environment.simulations_per_step = 2
        for _ in range(10):
            self.environment.step()
        self.assertTrue(self.environment.idx_step == 20)

    def test_computeData(self):
        # Compute input
        self.environment.computeInput()
        self.assertTrue(type(self.environment.getInput()) == np.ndarray)
        # Compute output
        self.environment.computeOutput()
        self.assertTrue(type(self.environment.getOutput()) == np.ndarray)
