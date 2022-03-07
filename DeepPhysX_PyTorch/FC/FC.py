from torch.nn import Sequential, PReLU, Linear
from DeepPhysX_PyTorch.Network.TorchNetwork import TorchNetwork


class FC(TorchNetwork):

    def __init__(self, config):
        TorchNetwork.__init__(self, config)
        # Convert biases to a List if not already one
        biases = None
        if isinstance(config.biases, list):
            if len(config.biases) != len(self.config.dim_layers) - 1:
                raise ValueError("Biases list length does not match layers count")
            biases = config.biases
        else:
            biases = [config.biases] * (len(self.config.dim_layers) - 1)
        # Init the layers
        self.layers = []
        for i, bias in enumerate(biases):
            self.layers.append(Linear(self.config.dim_layers[i], self.config.dim_layers[i+1], bias))
            self.layers.append(PReLU(num_parameters=self.config.dim_layers[i+1]))
        self.layers = self.layers[:-1]
        self.linear = Sequential(*self.layers)

    def forward(self, x):
        res = self.linear(x.view(x.shape[0], -1)).view(x.shape[0], -1, self.config.dim_output)
        return res

    def __str__(self):
        description = TorchNetwork.__str__(self)
        description += self.linear.__str__() + "\n"
        description += f"    Layers dimensions: {self.config.dim_layers}\n"
        description += f"    Output dimension: {self.config.dim_output}\n"
        return description
