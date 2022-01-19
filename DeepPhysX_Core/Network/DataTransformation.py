from typing import Callable, Any, Union, Tuple
import numpy
import torch

DataContainer = Union[numpy.ndarray, torch.Tensor]


class DataTransformation:

    def __init__(self, network_config):
        """
        DataTransformation is dedicated to data operations before and after network predictions.

        :param BaseNetworkConfig network_config: Specialisation containing the parameters of the network manager
        """

        self.network_config: Any = network_config  # BaseNetworkConfig

    @staticmethod
    def check_type(func: Callable[[Any, Any], Any]):
        def inner(self, *args):
            for data in args:
                if data is not None and type(data) != self.data_type:
                    raise TypeError(f"[{self.name}] Wrong data type: {self.data_type} required, get {type(data)}")
            return func(self, *args)

        return inner

    def transform_before_prediction(self, data_in: DataContainer) -> DataContainer:
        """
        Apply data operations before network's prediction.

        :param data_in: Input data
        :return: Transformed input data
        """
        return data_in

    def transform_before_loss(self, data_out: DataContainer, data_gt: DataContainer = None) -> Tuple[DataContainer, DataContainer]:
        """
        Apply data operations between network's prediction and loss computation.

        :param data_out: Prediction data
        :param data_gt: Ground truth data
        :return: Transformed prediction data, transformed ground truth data
        """
        return data_out, data_gt

    def transform_before_apply(self, data_out: DataContainer) -> DataContainer:
        """
        Apply data operations between loss computation and prediction apply in environment.

        :param data_out: Prediction data
        :return: Transformed prediction data
        """
        return data_out

    def __str__(self) -> str:
        description = "\n"
        description += f"  {self.__class__.__name__}\n"
        description += f"    Data type: {DataContainer.__repr__()}\n"
        description += f"    Transformation before prediction: Identity\n"
        description += f"    Transformation before loss: Identity\n"
        description += f"    Transformation before apply: Identity\n"
        return description
