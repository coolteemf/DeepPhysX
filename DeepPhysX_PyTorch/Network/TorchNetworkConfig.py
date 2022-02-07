from typing import Any, Optional, Type

from DeepPhysX_Core.Network.BaseNetworkConfig import BaseNetworkConfig
from DeepPhysX_PyTorch.Network.TorchDataTransformation import TorchDataTransformation
from DeepPhysX_PyTorch.Network.TorchNetwork import TorchNetwork
from DeepPhysX_PyTorch.Network.TorchOptimization import TorchOptimization


class TorchNetworkConfig(BaseNetworkConfig):

    def __init__(self,
                 network_class: Type[TorchNetwork] = TorchNetwork,
                 optimization_class: Type[TorchOptimization] = TorchOptimization,
                 data_transformation_class: Type[TorchDataTransformation] = TorchDataTransformation,
                 network_dir: str = None,
                 network_name: str = 'TorchNetwork',
                 network_type: str = 'TorchNetwork',
                 which_network: int = 0,
                 save_each_epoch: bool = False,
                 loss: Any = None,
                 lr: Optional[float] = None,
                 optimizer: Any = None):

        BaseNetworkConfig.__init__(self,
                                   network_class=network_class,
                                   optimization_class=optimization_class,
                                   data_transformation_class=data_transformation_class,
                                   network_dir=network_dir,
                                   network_name=network_name,
                                   network_type=network_type,
                                   which_network=which_network,
                                   save_each_epoch=save_each_epoch,
                                   loss=loss,
                                   lr=lr,
                                   optimizer=optimizer)

        # Change default config values for network only (configs for optimization and data_transformation are the same)
        self.network_config = self.make_config(config_name='network_config',
                                               network_name=network_name,
                                               network_type=network_type)
