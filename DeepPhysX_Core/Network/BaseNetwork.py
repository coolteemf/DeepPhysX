from typing import Union, Dict
import numpy
import torch

DataContainer = Union[numpy.ndarray, torch.Tensor]

class BaseNetwork:

    def __init__(self, config):
        """
        BaseNetwork is a network class to compute predictions from input data according to actual state.

        :param config: BaseNetworkConfig.BaseNetwork.Properties class containing BaseNetwork parameters
        """
        # Config
        self.device = None
        self.config = config

    def predict(self, x: DataContainer) -> DataContainer:
        return self.forward(x)

    def forward(self, x: DataContainer) -> DataContainer:
        raise NotImplementedError

    def set_train(self) -> None:
        raise NotImplementedError

    def set_eval(self) -> None:
        raise NotImplementedError

    def set_device(self) -> None:
        raise NotImplementedError

    def load_parameters(self, path: str) -> None:
        raise NotImplementedError

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def save_parameters(self, path) -> None:
        raise NotImplementedError

    def nb_parameters(self) -> None:
        raise NotImplementedError

    def transform_from_numpy(self, x: DataContainer, grad: bool = True) -> DataContainer:
        raise NotImplementedError

    def transform_to_numpy(self, x: DataContainer) -> DataContainer:
        raise NotImplementedError

    def __str__(self) -> str:
        """
        :return: String containing information about the BaseNetwork object
        """
        description = "\n"
        description += f"  {self.__class__.__name__}\n"
        description += f"    Name: {self.config.network_name}\n"
        description += f"    Type: {self.config.network_type}\n"
        description += f"    Number of parameters: {self.nb_parameters()}\n"
        description += f"    Estimated size: {self.nb_parameters() * 32 * 1.25e-10} Go\n"
        return description
