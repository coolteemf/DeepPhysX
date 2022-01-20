from typing import Any, List, Optional

from DeepPhysX_PyTorch.Network.TorchNetworkConfig import TorchNetworkConfig
from DeepPhysX_PyTorch.Network.TorchOptimization import TorchOptimization
from DeepPhysX_PyTorch.UNet.UnetDataTransformation import UnetDataTransformation
from DeepPhysX_PyTorch.UNet.UNet import UNet
from dataclasses import dataclass


class UNetConfig(TorchNetworkConfig):
    @dataclass
    class UNetProperties(TorchNetworkConfig.TorchNetworkProperties):
        input_size: List[int]
        nb_dims: int
        nb_input_channels: int
        nb_first_layer_channels: int
        nb_output_channels: int
        nb_steps: int
        two_sublayers: bool
        border_mode: str
        skip_merge: bool
        data_scale: float

    def __init__(self,
                 optimization_class: TorchOptimization = TorchOptimization,
                 data_transformation_class: UnetDataTransformation = UnetDataTransformation,
                 network_dir: str = None,
                 network_name: str = "UNetName",
                 which_network: int = 0,
                 save_each_epoch: bool = False,
                 loss: Any = None,
                 lr: Optional[float] = None,
                 optimizer: Any = None,
                 input_size: List[int] = None,
                 nb_dims: int = 3,
                 nb_input_channels: int = 1,
                 nb_first_layer_channels: int = 64,
                 nb_output_channels: int = 3,
                 nb_steps: int = 3,
                 two_sublayers: bool = True,
                 border_mode: str = 'valid',
                 skip_merge: bool = False,
                 data_scale: float = 1.):

        TorchNetworkConfig.__init__(self,
                                    network_class=UNet,
                                    optimization_class=optimization_class,
                                    data_transformation_class=data_transformation_class,
                                    network_dir=network_dir,
                                    network_name=network_name,
                                    network_type='UNet',
                                    which_network=which_network,
                                    save_each_epoch=save_each_epoch,
                                    loss=loss,
                                    lr=lr,
                                    optimizer=optimizer)

        if border_mode not in ['valid', 'same']:
            raise ValueError(
                    "[UNET_CONFIG] Border mode not in ['valid', 'same'], unknown value {}.".format(border_mode))
        if nb_dims not in [2, 3]:
            raise ValueError("[UNET_CONFIG] Nb of dimensions must be 2 or 3.")

        self.network_config = self.UNetProperties(network_name=network_name,
                                                  network_type='UNet',
                                                  input_size=input_size,
                                                  nb_dims=nb_dims,
                                                  nb_input_channels=nb_input_channels,
                                                  nb_first_layer_channels=nb_first_layer_channels,
                                                  nb_output_channels=nb_output_channels,
                                                  nb_steps=nb_steps,
                                                  two_sublayers=two_sublayers,
                                                  border_mode=border_mode,
                                                  skip_merge=skip_merge,
                                                  data_scale=data_scale)
