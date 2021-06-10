from DeepPhysX_PyTorch.Network.TorchNetworkConfig import TorchNetworkConfig
from DeepPhysX_PyTorch.FC.FC import FC
from dataclasses import dataclass


class FCConfig(TorchNetworkConfig):

    @dataclass
    class FCProperties(TorchNetworkConfig.TorchNetworkProperties):
        dim_output: int
        dim_layers: list

    def __init__(self, network_dir=None, network_name="FCName", save_each_epoch=False, which_network=0,
                 loss=None, lr=None, optimizer=None, dim_output=None, dim_layers=None):

        TorchNetworkConfig.__init__(self, network_class=FC, network_dir=network_dir, network_name=network_name,
                                    network_type='FC', save_each_epoch=save_each_epoch,
                                    which_network=which_network, loss=loss, lr=lr, optimizer=optimizer)

        self.network_config = self.FCProperties(network_name=network_name, network_type='UNet', dim_output=dim_output,
                                                dim_layers=dim_layers)
