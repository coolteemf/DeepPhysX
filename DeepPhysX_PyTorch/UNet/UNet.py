import torch
import torch.nn as nn

from DeepPhysX_PyTorch.Network.TorchNetwork import TorchNetwork
from DeepPhysX_PyTorch.EncoderDecoder.EncoderDecoder import EncoderDecoder
from .utils import crop_and_merge


class UNet(TorchNetwork):

    class UNetLayer(nn.Module):

        def __init__(self, nb_input_channels, nb_output_channels, config):
            super().__init__()
            self.nbInputChannels = nb_input_channels
            self.nbOutputChannels = nb_output_channels
            conv = nn.Conv2d if config.nb_dims == 2 else nn.Conv3d
            padding = 0 if config.border_mode == 'valid' else 1
            layers = [conv(nb_input_channels, nb_output_channels, kernel_size=3, padding=padding),
                      nn.BatchNorm3d(nb_output_channels),
                      nn.ReLU()]
            if config.two_sublayers:
                layers = layers + [conv(nb_output_channels, nb_output_channels, kernel_size=3, padding=padding),
                                   nn.BatchNorm3d(nb_output_channels),
                                   nn.ReLU()]
            self.unetLayer = nn.Sequential(*layers)

        def forward(self, x):
            return self.unetLayer(x)

    def __init__(self, config):
        TorchNetwork.__init__(self, config)

        self.maxPool = nn.MaxPool2d(2) if config.nb_dims == 2 else nn.MaxPool3d(2)
        last_conv_layer = nn.Conv2d if config.nb_dims == 2 else nn.Conv3d
        up_conv_layer = nn.ConvTranspose2d if config.nb_dims == 2 else nn.ConvTranspose3d

        in_count = config.first_layer_channels
        self.skipMerge = config.skip_merge
        # Down layers
        layers = [self.UNetLayer(config.nb_input_channels, in_count, config),
                  *[self.UNetLayer(in_count * 2 ** (i - 1), in_count * 2 ** i, config) for i in range(1, config.steps + 1)]]
        # Up layers
        layers = [*layers,
                  *[(up_conv_layer(in_count * 2 ** (i + 1), in_count * 2 ** i, kernel_size=2, stride=2),
                     self.UNetLayer(in_count * 2 ** (i + 1), in_count * 2 ** i, config)) for i in
                    range(config.steps - 1, -1, -1)],
                  last_conv_layer(in_count, config.nb_classes, 1)]
        self.architecture = EncoderDecoder(layers=layers, nb_encoding_layers=config.steps + 1)
        self.down = self.architecture.setupEncoder()
        self.middle = self.architecture.executeSequential()
        self.up = self.architecture.setupDecoder()
        self.finalLayer = self.architecture.decoder[-1]

    def forward(self, x):
        # Down layers
        down_outputs = [self.architecture.encoder[0](x)]
        for unet_layer in self.architecture.encoder[1:]:
            down_outputs.append(unet_layer(self.maxPool(down_outputs[-1])))
        # feature_maps = down_outputs.copy()
        x = down_outputs[-1]
        for (up_conv_layer, unet_layer), down_output in zip(self.architecture.decoder[:-1], down_outputs[-2::-1]):
            same_level_down_output = torch.zeros_like(down_output) if self.skipMerge else down_output
            x = unet_layer(crop_and_merge(same_level_down_output, up_conv_layer(x)))
            # feature_maps.append(x)
        # feature_maps.append(self.architecture.decoder[-1](x))
        # return feature_maps[-1]
        return self.architecture.decoder[-1](x)
