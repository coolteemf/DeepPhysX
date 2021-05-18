from DeepPhysX_PyTorch.Network.PyTorchBaseNetworkConfig import PyTorchBaseNetworkConfig
from .FC import FC
from dataclasses import dataclass


class FCConfig(PyTorchBaseNetworkConfig):

    @dataclass
    class FCProperties(PyTorchBaseNetworkConfig.PyTorchBaseNetworkProperties):
        dim_output: int
        dim_layers: list

    def __init__(self, network_name="FCName", loss=None, lr=None, optimizer=None,
                 network_dir=None, save_each_epoch=None, which_network=None,
                 dim_output=None, dim_layers=None):

        PyTorchBaseNetworkConfig.__init__(self, network_class=FC, network_name=network_name,
                                          network_type='FC', loss=loss, lr=lr, optimizer=optimizer,
                                          network_dir=network_dir, save_each_epoch=save_each_epoch,
                                          which_network=which_network)

        self.networkConfig = self.FCProperties(network_name=network_name, network_type='UNet', dim_output=dim_output,
                                               dim_layers=dim_layers)

