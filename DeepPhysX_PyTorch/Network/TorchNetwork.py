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
        torch.nn.Module.__init__(self)
        BaseNetwork.__init__(self, config)

    def forward(self, x: DataContainer) -> DataContainer:
        raise NotImplementedError

    def set_train(self) -> None:
        self.train()

    def set_eval(self) -> None:
        self.eval()

    def set_device(self) -> None:
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            # Garbage collector run
            gc.collect()
            torch.cuda.empty_cache()
        else:
            self.device = torch.device('cpu')
            torch.set_num_threads(psutil.cpu_count(logical=True) - 1)
        self.to(self.device)
        print("[{}]: Device is {}".format(self.__class__.__name__, self.device))

    def load_parameters(self, path: str) -> None:
        self.load_state_dict(torch.load(path, map_location=self.device))

    def get_parameters(self) -> Dict[str, Tensor]:
        return self.state_dict()

    def save_parameters(self, path: str) -> None:
        path = path + '.pth'
        torch.save(self.state_dict(), path)

    def nb_parameters(self) -> None:
        return sum(p.numel() for p in self.parameters())

    def transform_from_numpy(self, x: DataContainer, grad: bool = True) -> DataContainer:
        x = torch.as_tensor(x, dtype=torch.float, device=self.device)
        if grad:
            x.requires_grad_()
        return x

    def transform_to_numpy(self, x: DataContainer) -> DataContainer:
        return x.cpu().detach().numpy()

    def print_architecture(self, architecture) -> str:
        lines = architecture.splitlines()
        architecture = ''
        for line in lines:
            architecture += '\n      ' + line
        return architecture

    def __str__(self):
        description = BaseNetwork.__str__(self)
        description += f"    Device: {self.device}\n"
        return description
