import unittest
import torch

from DeepPhysX_PyTorch.Network.TorchNetworkConfig import TorchNetworkConfig
from DeepPhysX_PyTorch.Network.TorchNetwork import TorchNetwork
from DeepPhysX_PyTorch.Network.TorchOptimization import TorchOptimization


class TestTorchNetworkConfig(unittest.TestCase):

    def setUp(self):
        pass

    def test_init(self):
        with self.assertRaises(TypeError):
            TorchNetworkConfig(network_dir=0)
            TorchNetworkConfig(network_name=None)
            TorchNetworkConfig(network_type=None)
            TorchNetworkConfig(which_network=None)
            TorchNetworkConfig(save_each_epoch='True')

    def test_createNetwork(self):
        # Bad
        config = TorchNetworkConfig(network_class=TorchNetworkConfig)
        self.assertRaises(TypeError, config.createNetwork)
        # Good
        config = TorchNetworkConfig(network_class=TorchNetwork)
        self.assertIsInstance(config.createNetwork(), TorchNetwork)
        self.assertIsInstance(config.createNetwork(), torch.nn.Module)

    def test_createOptimization(self):
        # Bad
        config = TorchNetworkConfig(optimization_class=TorchNetworkConfig)
        self.assertRaises(TypeError, config.createOptimization)
        # Good
        config = TorchNetworkConfig(optimization_class=TorchOptimization)
        self.assertIsInstance(config.createOptimization(), TorchOptimization)


class TestTorchNetwork(unittest.TestCase):

    def setUp(self):
        config = TorchNetworkConfig()
        self.network = config.createNetwork()

    def test_notImplemented(self):
        with self.assertRaises(NotImplementedError):
            self.network.forward(None)

    def test_setTrainingMode(self):
        self.network.setTrain()
        self.assertTrue(self.network.training == True)
        self.network.setEval()
        self.assertTrue(self.network.training == False)

    def test_setDevice(self):
        self.network.setDevice()
        self.assertTrue(str(self.network.device) in ['cpu', 'cuda'])

    def test_parameters(self):
        # todo: nbParameters, getParameters, saveParameters, loadParameters
        pass

    def test_transformNumpy(self):
        # todo: transformFromNumpy, transformToNumpy
        pass
