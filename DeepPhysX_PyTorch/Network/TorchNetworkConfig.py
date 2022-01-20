from DeepPhysX_Core.Network.BaseNetworkConfig import *
from DeepPhysX_Core.Network.DataTransformation import DataTransformation
from DeepPhysX_PyTorch.Network.TorchNetwork import TorchNetwork
from DeepPhysX_PyTorch.Network.TorchOptimization import TorchOptimization
from dataclasses import dataclass


class TorchNetworkConfig(BaseNetworkConfig):
    @dataclass
    class TorchNetworkProperties(BaseNetworkConfig.BaseNetworkProperties):
        pass

    @dataclass
    class TorchOptimizationProperties(BaseOptimization.BaseOptimizationProperties):
        pass

    def __init__(self,
                 network_class: TorchNetwork = TorchNetwork,
                 optimization_class: TorchOptimization = TorchOptimization,
                 data_transformation_class: DataTransformation = DataTransformation,
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

        self.network_config: Any = self.TorchNetworkProperties(network_name=network_name, network_type=network_type)
        self.optimization_config: Any = self.TorchOptimizationProperties(loss=loss, lr=lr, optimizer=optimizer)
