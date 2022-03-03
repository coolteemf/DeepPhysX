from typing import Any, Optional, Type, Union, List

from DeepPhysX_PyTorch.Network.TorchNetworkConfig import TorchNetworkConfig, TorchDataTransformation, TorchOptimization
from DeepPhysX_PyTorch.FC.FC import FC


class FCConfig(TorchNetworkConfig):

    def __init__(self,
                 network_dir: str = None,
                 network_name: str = "FCNetwork",
                 optimization_class: Type[TorchOptimization] = TorchOptimization,
                 save_each_epoch: bool = False,
                 which_network: int = 0,
                 data_transformation_class: Type[TorchDataTransformation] = TorchDataTransformation,
                 lr: Optional[float] = None,
                 require_training_stuff: bool = True,
                 loss: Any = None,
                 optimizer: Any = None,
                 dim_output: int = 0,
                 dim_layers: list = None,
                 biases: Union[List[bool], bool] = True):

        TorchNetworkConfig.__init__(self,
                                    network_class=FC,
                                    network_dir=network_dir,
                                    network_name=network_name,
                                    network_type='FC',
                                    save_each_epoch=save_each_epoch,
                                    optimization_class=optimization_class,
                                    which_network=which_network,
                                    data_transformation_class=data_transformation_class,
                                    require_training_stuff=require_training_stuff,
                                    lr=lr,
                                    loss=loss,
                                    optimizer=optimizer)

        # Check FC variables
        if dim_output is not None and type(dim_output) != int:
            raise TypeError(f"[{self.__class__.__name__}] Wrong 'dim_output' type: int required, get "
                            f"{type(dim_output)}")
        dim_layers = dim_layers if dim_layers else []
        if type(dim_layers) != list:
            raise TypeError(f"[{self.__class__.__name__}] Wrong 'dim_layers' type: list required, get "
                            f"{type(dim_layers)}")

        self.network_config = self.make_config(config_name='network_config',
                                               network_name=network_name,
                                               network_type='FC',
                                               dim_output=dim_output,
                                               dim_layers=dim_layers,
                                               biases=biases)
