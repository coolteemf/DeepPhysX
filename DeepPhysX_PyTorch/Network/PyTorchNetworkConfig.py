from DeepPhysX.Network.NetworkConfig import NetworkConfig
from .PyTorchNetworkOptimization import PyTorchNetworkOptimization


class PyTorchNetworkConfig(NetworkConfig):

    def __init__(self, network_name, network_type, network_dir=None, loss=None, lr=None, optimizer=None,
                 save_each_epoch=None, which_network=1):
        NetworkConfig.__init__(self, network_name, network_type, network_dir=network_dir, loss=loss, lr=lr,
                               optimizer=optimizer, save_each_epoch=save_each_epoch, which_network=which_network)
        self.optimization = PyTorchNetworkOptimization
