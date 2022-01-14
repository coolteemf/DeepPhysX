from typing import Union, Tuple
import numpy
import torch

from DeepPhysX_Core.Network.DataTransformation import DataTransformation

DataContainer = Union[numpy.ndarray, torch.Tensor]


class TorchDataTransformation(DataTransformation):

    def __init__(self, network_config):
        super().__init__(network_config)

    @DataTransformation.check_type
    def transform_before_prediction(self, data_in: DataContainer) -> DataContainer:
        return data_in

    @DataTransformation.check_type
    def transform_before_loss(self, data_out: DataContainer, data_gt: DataContainer = None) -> Tuple[DataContainer, DataContainer]:
        return data_out, data_gt

    @DataTransformation.check_type
    def transform_before_apply(self, data_out: DataContainer) -> DataContainer:
        return data_out

    def __str__(self) -> str:
        description = "\n"
        description += f"  {self.__class__.__name__}\n"
        description += f"    Data type: {DataContainer.__repr__()}\n"
        description += f"    Transformation before prediction: Identity\n"
        description += f"    Transformation before loss: Identity\n"
        description += f"    Transformation before apply: Identity\n"
        return description
