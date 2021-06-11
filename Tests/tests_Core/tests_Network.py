import os
import shutil
import unittest

import numpy as np

from DeepPhysX.Network.BaseNetworkConfig import BaseNetworkConfig
from DeepPhysX.Network.BaseNetwork import BaseNetwork
from DeepPhysX.Network.BaseOptimization import BaseOptimization
from DeepPhysX.Manager.NetworkManager import NetworkManager
from Tests.tests_Core.numpy_Network import NumpyNetwork, NumpyOptimisation


class TestBaseNetworkConfig(unittest.TestCase):

    def setUp(self):
        pass

    def test_init(self):
        with self.assertRaises(TypeError):
            BaseNetworkConfig(network_dir=0)
            BaseNetworkConfig(network_name=None)
            BaseNetworkConfig(network_type=None)
            BaseNetworkConfig(which_network=None)
            BaseNetworkConfig(save_each_epoch='True')

    def test_createNetwork(self):
        # Bad
        config = BaseNetworkConfig(network_class=BaseNetworkConfig)
        self.assertRaises(TypeError, config.createNetwork)
        # Good
        config = BaseNetworkConfig(network_class=BaseNetwork)
        self.assertIsInstance(config.createNetwork(), BaseNetwork)

    def test_createOptimization(self):
        # Bad
        config = BaseNetworkConfig(optimization_class=BaseNetworkConfig)
        self.assertRaises(TypeError, config.createOptimization)
        # Good
        config = BaseNetworkConfig(optimization_class=BaseOptimization)
        self.assertIsInstance(config.createOptimization(), BaseOptimization)


class TestBaseNetwork(unittest.TestCase):

    def setUp(self):
        config = BaseNetworkConfig()
        self.network = config.createNetwork()

    def test_notImplemented(self):
        with self.assertRaises(NotImplementedError):
            self.network.forward(None)
            self.network.setTrain()
            self.network.setEval()
            self.network.setDevice()
            self.network.loadParameters()
            self.network.getParameters()
            self.network.saveParameters()
            self.network.nbParameters()
            self.network.transformFromNumpy()
            self.network.transformToNumpy()


class TestBaseOptimization(unittest.TestCase):

    def setUp(self):
        config = BaseNetworkConfig()
        self.optimization = config.createOptimization()

    def test_notImplemented(self):
        with self.assertRaises(NotImplementedError):
            self.optimization.setLoss()
            self.optimization.computeLoss(None, None)
            self.optimization.setOptimizer(None)
            self.optimization.optimize()


class TestNetworkManager(unittest.TestCase):

    def setUp(self):
        self.network_config = BaseNetworkConfig(network_class=NumpyNetwork, optimization_class=NumpyOptimisation,
                                                lr=0.1)
        self.session_dir = os.path.join(os.getcwd(), 'session')

    def test_init(self):
        with self.assertRaises(TypeError):
            NetworkManager(self.network_config, session_name=None)
            NetworkManager(self.network_config, session_dir=0)
            NetworkManager(self.network_config, train='True')
        with self.assertRaises(ValueError):
            NetworkManager(self.network_config, train=True)

    def test_setNetwork(self):
        self.network_config.training_stuff = True
        manager = NetworkManager(network_config=self.network_config, session_dir=self.session_dir,
                                 session_name='session')
        self.assertIsInstance(manager.network, BaseNetwork)
        self.assertIsInstance(manager.optimization, BaseOptimization)
        self.assertTrue(os.path.exists(manager.session_dir))
        self.assertTrue(os.path.exists(manager.network_dir))

    def test_setData(self):
        self.network_config.training_stuff = True
        manager = NetworkManager(network_config=self.network_config, session_dir=self.session_dir)
        data = {'in': np.array([0.5]), 'out': np.array([0.5])}
        manager.setData(data)
        self.assertTrue((manager.network.input == data['in']).all())
        self.assertTrue((manager.network.ground_truth == data['out']).all())

    def test_optimizeNetwork(self):
        self.network_config.training_stuff = True
        manager = NetworkManager(network_config=self.network_config, session_dir=self.session_dir)
        x = {'in': np.array([0.5]), 'out': np.array([0.5])}
        manager.setData(x)
        loss_values = []
        for _ in range(10):
            loss_values.append(manager.optimizeNetwork(*manager.computePrediction()))
            if len(loss_values) > 1:
                self.assertTrue(loss_values[-2] > loss_values[-1])

    def test_saveNetwork(self):
        self.network_config.training_stuff = True
        self.network_config.save_each_epoch = True
        manager = NetworkManager(network_config=self.network_config, session_dir=self.session_dir)
        parameters = manager.network.getParameters()
        for _ in range(3):
            manager.saveNetwork()
        networks = [file for file in os.listdir(manager.network_dir)]
        self.assertTrue(len(networks) == 3)
        manager.close()
        networks = [file for file in os.listdir(manager.network_dir)]
        self.assertTrue(len(networks) == 4)

    def tearDown(self):
        if os.path.exists(self.session_dir):
            shutil.rmtree(self.session_dir)
