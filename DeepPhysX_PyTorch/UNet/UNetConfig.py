from DeepPhysX_PyTorch.Network.PyTorchBaseNetworkConfig import PyTorchBaseNetworkConfig
from .UNet import UNet
from dataclasses import dataclass


class UNetConfig(PyTorchBaseNetworkConfig):

    @dataclass
    class UNetProperties(PyTorchBaseNetworkConfig.PyTorchBaseNetworkProperties):
        nb_dims: int
        first_layer_channels: int
        border_mode: str
        two_sublayers: bool
        nb_input_channels: int
        steps: int
        nb_classes: int
        skip_merge: bool

    def __init__(self, network_name="UNetName", loss=None, lr=None, optimizer=None,
                 network_dir=None, save_each_epoch=None, which_network=None,
                 steps=4, first_layer_channels=64, nb_classes=2, nb_input_channels=1, two_sublayers=True,
                 nb_dims=2, border_mode='valid', skip_merge=False):

        PyTorchBaseNetworkConfig.__init__(self, network_class=UNet, network_name=network_name,
                                          network_type='UNet', loss=loss, lr=lr, optimizer=optimizer,
                                          network_dir=network_dir, save_each_epoch=save_each_epoch,
                                          which_network=which_network)

        if border_mode not in ['valid', 'same']:
            raise ValueError(
                "[UNET_CONFIG] Border mode not in ['valid', 'same'], unknown value {}.".format(border_mode))
        if nb_dims not in [2, 3]:
            raise ValueError("[UNET_CONFIG] Nb of dimensions must be 2 or 3.")

        self.networkConfig = self.UNetProperties(network_name=network_name, network_type='UNet', nb_dims=nb_dims,
                                                 first_layer_channels=first_layer_channels, border_mode=border_mode,
                                                 two_sublayers=two_sublayers, nb_input_channels=nb_input_channels,
                                                 steps=steps, nb_classes=nb_classes, skip_merge=skip_merge)
        border = 4 if two_sublayers else 2
        if border_mode == 'same':
            border = 0
        self.first_step = lambda x: x - border
        self.rev_first_step = lambda x: x + border
        self.down_step = lambda x: (x - 1) // 2 + 1 - border
        self.rev_down_step = lambda x: (x + border) * 2
        self.up_step = lambda x: (x * 2) - border
        self.rev_up_step = lambda x: (x + border - 1) // 2 + 1
