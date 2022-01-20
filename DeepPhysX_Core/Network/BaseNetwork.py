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

    def predict(self, input_data: DataContainer) -> DataContainer:
        """Calls forward"""
        return self.forward(input_data)

    def forward(self, input_data: DataContainer) -> DataContainer:
        """Gives input_data as raw input to the neural network"""
        raise NotImplementedError

    def set_train(self) -> None:
        """Network is now in train mode (compute gradient)"""
        raise NotImplementedError

    def set_eval(self) -> None:
        """Network is now in eval mode (does not compute gradient)"""
        raise NotImplementedError

    def set_device(self) -> None:
        """Update default device"""
        raise NotImplementedError

    def load_parameters(self, path: str) -> None:
        """Load network parameter from path"""
        raise NotImplementedError

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Return network parameter"""
        raise NotImplementedError

    def save_parameters(self, path) -> None:
        """Saves the network parameter to the path location"""
        raise NotImplementedError

    def nb_parameters(self) -> int:
        """Return the number of parameters of the network"""
        raise NotImplementedError

    def transform_from_numpy(self, data: numpy.ndarray, grad: bool = True) -> DataContainer:
        """Transform and cast data data from numpy to the desired tensor type"""
        raise NotImplementedError

    def transform_to_numpy(self, data: DataContainer) -> numpy.ndarray:
        """Transform and cast data from tensor type to numpy"""
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
