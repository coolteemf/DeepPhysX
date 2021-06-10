from DeepPhysX.Network.BaseNetworkConfig import BaseNetworkConfig
from DeepPhysX_PyTorch.Network.TorchNetwork import TorchNetwork
from DeepPhysX_PyTorch.Network.TorchOptimization import TorchOptimization
from dataclasses import dataclass


class TorchNetworkConfig(BaseNetworkConfig):

    @dataclass
    class TorchNetworkProperties(BaseNetworkConfig.BaseNetworkProperties):
        pass

    @dataclass
    class TorchOptimizationProperties(BaseNetworkConfig.BaseOptimizationProperties):
        pass

    def __init__(self, network_class=TorchNetwork, optimization_class=TorchOptimization, network_dir=None,
                 network_name='TorchNetwork', network_type='TorchNetwork', which_network=0, save_each_epoch=False,
                 loss=None, lr=None, optimizer=None):

        BaseNetworkConfig.__init__(self, network_class=network_class, optimization_class=optimization_class,
                                   network_dir=network_dir, network_name=network_name, network_type=network_type,
                                   save_each_epoch=save_each_epoch, which_network=which_network,
                                   loss=loss, lr=lr, optimizer=optimizer)

        self.network_config = self.TorchNetworkProperties(network_name=network_name, network_type=network_type)
        self.optimization_config = self.TorchOptimizationProperties(loss=loss, lr=lr, optimizer=optimizer)
