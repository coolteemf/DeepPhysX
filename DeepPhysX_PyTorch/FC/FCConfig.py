from typing import Any, Optional

from DeepPhysX_PyTorch.Network.TorchNetworkConfig import TorchNetworkConfig, DataTransformation, TorchOptimization
from DeepPhysX_PyTorch.FC.FC import FC
from dataclasses import dataclass


class FCConfig(TorchNetworkConfig):

    @dataclass
    class FCProperties(TorchNetworkConfig.TorchNetworkProperties):
        dim_output: int
        dim_layers: list

    def __init__(self,
                 network_dir: str =None,
                 network_name: str ="FCName",
                 optimization_class: TorchOptimization = TorchOptimization,
                 save_each_epoch: bool =False,
                 which_network: int =0,
                 data_transformation_class: DataTransformation =DataTransformation,
                 loss: Any =None,
                 lr: Optional[float] =None,
                 optimizer: Any =None,
                 dim_output: Optional[float] =None,
                 dim_layers: Optional[float] =None):

        TorchNetworkConfig.__init__(self, network_class=FC, network_dir=network_dir, network_name=network_name,
                                    network_type='FC', save_each_epoch=save_each_epoch, optimization_class=optimization_class,
                                    which_network=which_network, data_transformation_class=data_transformation_class,
                                    loss=loss, lr=lr, optimizer=optimizer)

        self.network_config = self.FCProperties(network_name=network_name, network_type='UNet', dim_output=dim_output,
                                                dim_layers=dim_layers)
