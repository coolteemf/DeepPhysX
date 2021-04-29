from DeepPhysX.Network.BaseNetworkConfig import BaseNetworkConfig
from .PyTorchBaseNetwork import
from .PyTorchNetworkOptimization import PyTorchBaseOptimization


class PyTorchBaseNetworkConfig(BaseNetworkConfig):

    class NetworkProperties(BaseNetworkConfig.NetworkProperties):
        def __init__(self, network_name, network_type):
            super().__init__(network_name, network_type)

    class OptimizationProperties(BaseNetworkConfig.OptimizationProperties):
        def __init__(self, loss, lr, optimizer):
            super().__init__(loss, lr, optimizer)

    def __init__(self, network_class, network_name="", network_type="", loss=None, lr=None, optimizer=None,
                 network_dir=None, save_each_epoch=None, which_network=None):
        BaseNetworkConfig.__init__(self, network_class=network_class, network_name=network_name, network_type=network_type,
                                   loss=loss, lr=lr, optimizer=optimizer, network_dir=network_dir,
                                   save_each_epoch=save_each_epoch, which_network=which_network)
        self.optimization_class = PyTorchBaseOptimization
        self.descriptionName = "PYTORCH NetworkConfig"
