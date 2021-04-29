import torch

from DeepPhysX_PyTorch.Network.PyTorchBaseNetwork import PyTorchBaseNetwork


class MyNetwork(PyTorchBaseNetwork):

    def __init__(self, network_name="MyNetwork_Name", *args):
        PyTorchBaseNetwork.__init__(self, network_name=network_name, network_type="MyNetwork")
        self.fc1 = torch.nn.Linear(1, 10)
        self.fc2 = torch.nn.Linear(10, 10)
        self.fc3 = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return self.fc3(x)
