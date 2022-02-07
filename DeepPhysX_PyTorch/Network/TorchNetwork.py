from typing import Union, Dict
import torch
import numpy
import gc
import psutil
from torch import Tensor

from DeepPhysX_Core.Network.BaseNetwork import BaseNetwork

DataContainer = Union[numpy.ndarray, torch.Tensor]


class TorchNetwork(torch.nn.Module, BaseNetwork):

    def __init__(self, config):
        """
        TorchNetwork is a network class to compute predictions from input data according to actual state.

        :param config: BaseNetworkConfig.BaseNetwork.Properties class containing BaseNetwork parameters
        """
        torch.nn.Module.__init__(self)
        BaseNetwork.__init__(self, config)

    def forward(self, input_data: DataContainer) -> DataContainer:
        """Gives input_data as raw input to the neural network"""
        raise NotImplementedError

    def set_train(self) -> None:
        """Network is now in train mode (compute gradient)"""
        self.train()

    def set_eval(self) -> None:
        """Network is now in eval mode (does not compute gradient)"""
        self.eval()

    def set_device(self) -> None:
        """Update default device"""
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            # Garbage collector run
            gc.collect()
            torch.cuda.empty_cache()
        else:
            self.device = torch.device('cpu')
            torch.set_num_threads(psutil.cpu_count(logical=True) - 1)
        self.to(self.device)
        print(f"[{self.__class__.__name__}]: Device is {self.device}")

    def load_parameters(self, path: str) -> None:
        """Load network parameter from path"""
        self.load_state_dict(torch.load(path, map_location=self.device))

    def get_parameters(self) -> Dict[str, Tensor]:
        """Return network parameter"""
        return self.state_dict()

    def save_parameters(self, path: str) -> None:
        """Saves the network parameter to the path location"""
        path = path + '.pth'
        torch.save(self.state_dict(), path)

    def nb_parameters(self) -> int:
        """Return the number of parameters of the network"""
        return sum(p.numel() for p in self.parameters())

    def transform_from_numpy(self, x: DataContainer, grad: bool = True) -> DataContainer:
        """Transform and cast data data from numpy to the desired tensor type"""
        x = torch.as_tensor(x, dtype=torch.float, device=self.device)
        if grad:
            x.requires_grad_()
        return x

    def transform_to_numpy(self, x: DataContainer) -> DataContainer:
        """Transform and cast data from tensor type to numpy"""
        return x.cpu().detach().numpy()

    def print_architecture(self, architecture):
        lines = architecture.splitlines()
        architecture = ''
        for line in lines:
            architecture += '\n      ' + line
        return architecture

    def __str__(self):
        """
        :return: String containing information about the BaseNetwork object
        """
        description = BaseNetwork.__str__(self)
        description += f"    Device: {self.device}\n"
        return description
