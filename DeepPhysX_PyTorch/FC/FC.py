from typing import List, Union

import torch
from torch.nn import Sequential, PReLU, Module, Linear

from DeepPhysX_PyTorch.Network.TorchNetwork import TorchNetwork


class FCLayer(Module):

    def __init__(self,
                 nb_input_channels,
                 nb_output_channels):
        """Creates one fully connected layer of dimension nb_input_channels X nb_output_channels"""
        super().__init__()
        self.linear = Sequential(Linear(nb_input_channels, nb_output_channels, False))

    def forward(self, input_data):
        """Gives input_data as raw input to the neural network"""
        res = self.linear(input_data)
        return res


class FC(TorchNetwork):

    def __init__(self, config):
        """Creates a simple fully connected layers neural network"""
        TorchNetwork.__init__(self, config)

        self.layers: List[Union[FCLayer, PReLU]] = []
        for i in range(len(self.config.dim_layers) - 1):
            self.layers.append(FCLayer(nb_input_channels=self.config.dim_layers[i],
                                       nb_output_channels=self.config.dim_layers[i + 1]))
            self.layers.append(PReLU(num_parameters=self.config.dim_layers[i + 1]))
        self.linear = Sequential(*(self.layers[:-1]))

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Gives input_data as raw input to the neural network"""
        return self.linear(input_data.view(input_data.shape[0], -1)).view(input_data.shape[0], -1, self.config.dim_output)

    def __str__(self):
        description = TorchNetwork.__str__(self)
        description += f"    Layers dimensions: {self.config.dim_layers}\n"
        description += f"    Output dimension: {self.config.dim_output}\n"
        description += f"    Layers: "
        for layer in self.layers:
            description += self.print_architecture(str(layer))
        return description
