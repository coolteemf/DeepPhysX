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
        for i, (dim_in, dim_out) in enumerate(zip(self.config.dim_layers[0:-2], self.config.dim_layers[1:-1])):
            self.layers.append(Linear(dim_in, dim_out, biases[i]))
            self.layers.append(PReLU(num_parameters=dim_out))
        self.layers.append(Linear(self.config.dim_layers[-2],
                                  self.config.dim_layers[-1], biases[-1]))

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
