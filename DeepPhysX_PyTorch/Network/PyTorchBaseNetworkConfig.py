from DeepPhysX.Network.BaseNetworkConfig import BaseNetworkConfig
from .PyTorchBaseNetwork import PyTorchBaseNetwork
from .PyTorchBaseOptimization import PyTorchBaseOptimization
from dataclasses import dataclass


class PyTorchBaseNetworkConfig(BaseNetworkConfig):

    @dataclass
    class PyTorchBaseNetworkProperties(BaseNetworkConfig.BaseNetworkProperties):
        pass

    @dataclass
    class PyTorchBaseOptimizationProperties(BaseNetworkConfig.BaseOptimizationProperties):
        pass

    def __init__(self, network_class=PyTorchBaseNetwork, optimization_class=PyTorchBaseOptimization,
                 network_name="", network_type="", loss=None, lr=None, optimizer=None, network_dir=None,
                 save_each_epoch=None, which_network=None):
        BaseNetworkConfig.__init__(self, network_class=network_class, optimization_class=optimization_class,
                                   network_name=network_name, network_type=network_type,
                                   loss=loss, lr=lr, optimizer=optimizer, network_dir=network_dir,
                                   save_each_epoch=save_each_epoch, which_network=which_network)
        self.networkConfig = self.PyTorchBaseNetworkProperties(network_name=network_name, network_type=network_type)
        self.optimizationConfig = self.PyTorchBaseOptimizationProperties(loss=loss, lr=lr, optimizer=optimizer)
        self.descriptionName = "PYTORCH NetworkConfig"
