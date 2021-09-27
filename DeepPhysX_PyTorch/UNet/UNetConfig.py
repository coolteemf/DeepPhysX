import numpy as np

from DeepPhysX_PyTorch.Network.TorchNetworkConfig import TorchNetworkConfig
from DeepPhysX_PyTorch.UNet.UnetDataTransformation import UnetDataTransformation
from DeepPhysX_PyTorch.UNet.UNet import UNet
from dataclasses import dataclass


class UNetConfig(TorchNetworkConfig):
    @dataclass
    class UNetProperties(TorchNetworkConfig.TorchNetworkProperties):
        nb_dims: int
        first_layer_channels: int
        border_mode: str
        two_sublayers: bool
        nb_input_channels: int
        steps: int
        nb_classes: int
        skip_merge: bool
        grid_shape: list
        data_scale: float

    def __init__(self, network_name="UNetName", data_transformation_class=UnetDataTransformation,
                 loss=None, lr=None, optimizer=None,
                 network_dir=None, save_each_epoch=False, which_network=0,
                 steps=4, first_layer_channels=64, nb_classes=2, nb_input_channels=1, two_sublayers=True,
                 nb_dims=2, border_mode='valid', skip_merge=False, grid_shape=None, data_scale=1.):

        TorchNetworkConfig.__init__(self, network_class=UNet, network_name=network_name,
                                    network_type='UNet', data_transformation_class=data_transformation_class,
                                    loss=loss, lr=lr, optimizer=optimizer,
                                    network_dir=network_dir, save_each_epoch=save_each_epoch,
                                    which_network=which_network)

        if border_mode not in ['valid', 'same']:
            raise ValueError(
                "[UNET_CONFIG] Border mode not in ['valid', 'same'], unknown value {}.".format(border_mode))
        if nb_dims not in [2, 3]:
            raise ValueError("[UNET_CONFIG] Nb of dimensions must be 2 or 3.")

        self.network_config = self.UNetProperties(network_name=network_name, network_type='UNet', nb_dims=nb_dims,
                                                  first_layer_channels=first_layer_channels, border_mode=border_mode,
                                                  two_sublayers=two_sublayers, nb_input_channels=nb_input_channels,
                                                  steps=steps, nb_classes=nb_classes, skip_merge=skip_merge,
                                                  grid_shape=grid_shape, data_scale=data_scale)
        border = 4 if two_sublayers else 2
        if border_mode == 'same':
            border = 0
        self.first_step = lambda x: x - border
        self.rev_first_step = lambda x: x + border
        self.down_step = lambda x: (x - 1) // 2 + 1 - border
        self.rev_down_step = lambda x: (x + border) * 2
        self.up_step = lambda x: (x * 2) - border
        self.rev_up_step = lambda x: (x + border - 1) // 2 + 1

    def out_shape(self, in_shape):
        """
        Return the shape of the output given the shape of the input
        :param in_shape:
        :return:
        """
        shapes = self.features_map_shapes(in_shape)
        return shapes[-1][1:]

    def features_map_shapes(self, in_shape):

        def _feature_map_shapes():
            shape = np.asarray(in_shape)
            yield (self.network_config.nb_input_channels,) + tuple(shape)
            shape = self.first_step(shape)
            yield (self.network_config.first_layer_channels,) + tuple(shape)
            for i in range(self.network_config.steps):
                shape = self.down_step(shape)
                channels = self.network_config.first_layer_channels * 2 ** (i + 1)
                yield (channels,) + tuple(shape)
            for i in range(self.network_config.steps):
                shape = self.up_step(shape)
                channels = self.network_config.first_layer_channels * 2 ** (self.network_config.steps - i - 1)
                yield (channels,) + tuple(shape)
            yield (self.network_config.nb_classes,) + tuple(shape)

        return list(_feature_map_shapes())

    def out_step(self):
        _, out_shape0 = self.in_out_shape((0,))
        _, out_shape1 = self.in_out_shape((out_shape0[0] + 1,))
        return out_shape1[0] - out_shape0[0]

    def in_out_shape(self, out_shape_lower_bound, given_upper_bound=False):
        """
        Compute the best combination of input/output shapes given the desired lower bound for the shape of the output
        :param out_shape_lower_bound:
        :param given_upper_bound:
        :return:
        """
        if given_upper_bound:
            out_shape_upper_bound = out_shape_lower_bound
            out_step = self.out_step()
            out_shape_lower_bound = tuple(i - out_step + 1 for i in out_shape_upper_bound)
        shape = np.asarray(out_shape_lower_bound)
        for i in range(self.network_config.steps):
            shape = self.rev_up_step(shape)
        # Compute correct out shape from minimum shape
        out_shape = np.copy(shape)
        for i in range(self.network_config.steps):
            out_shape = self.up_step(out_shape)
        # Best input shape
        for i in range(self.network_config.steps):
            shape = self.rev_down_step(shape)
        shape = self.rev_first_step(shape)
        return tuple(shape), tuple(out_shape)

    def in_out_pad_widths(self, out_shape_lower_bound):
        in_shape, out_shape = self.in_out_shape(out_shape_lower_bound)
        in_pad_widths = [((sh_o - sh_i) // 2, (sh_o - sh_i - 1) // 2 + 1)
                         for sh_i, sh_o in zip(out_shape_lower_bound, in_shape)]
        out_pad_widths = [((sh_o - sh_i) // 2, (sh_o - sh_i - 1) // 2 + 1)
                          for sh_i, sh_o in zip(out_shape_lower_bound, out_shape)]
        return in_pad_widths, out_pad_widths
