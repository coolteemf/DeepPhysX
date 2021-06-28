import torch
import torch.nn as nn

from DeepPhysX_PyTorch.Network.TorchNetwork import TorchNetwork


class FC(TorchNetwork):

    class FCLayer(nn.Module):

        def __init__(self, nb_input_channels, nb_output_channels):
            super().__init__()
            self.nbInChannels = nb_input_channels
            self.nbOutChannels = nb_output_channels
            self.linear = nn.Sequential(nn.Linear(self.nbInChannels, self.nbOutChannels, False))

        def forward(self, x):
            res = self.linear(x)
            return res

    def __init__(self, config):
        TorchNetwork.__init__(self, config)

        self.layers = []
        for i in range(len(self.config.dim_layers) - 1):
            self.layers.append(self.FCLayer(nb_input_channels=self.config.dim_layers[i],
                                            nb_output_channels=self.config.dim_layers[i+1]))
            self.layers.append(nn.PReLU(num_parameters=self.config.dim_layers[i+1]))
        self.linear = nn.Sequential(*(self.layers[:-1]))

    def forward(self, x):
        # print("\nin\n", x)
        res = self.linear(x.view(x.shape[0], -1)).view(x.shape[0], -1, self.config.dim_output)
        return res
