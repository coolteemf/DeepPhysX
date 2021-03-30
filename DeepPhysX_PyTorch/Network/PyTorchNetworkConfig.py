from DeepPhysX.Network.NetworkConfig import NetworkConfig
from .PyTorchNetworkOptimization import PyTorchNetworkOptimization


class PyTorchNetworkConfig(NetworkConfig):

    def __init__(self, network_class, network_name="", network_type="", loss=None, lr=None, optimizer=None,
                 network_dir=None, save_each_epoch=None, which_network=None):
        NetworkConfig.__init__(self, network_class=network_class, network_name=network_name, network_type=network_type,
                               loss=loss, lr=lr, optimizer=optimizer, network_dir=network_dir,
                               save_each_epoch=save_each_epoch, which_network=which_network)
        self.optimization_class = PyTorchNetworkOptimization
        self.descriptionName = "PYTORCH NetworkConfig"
