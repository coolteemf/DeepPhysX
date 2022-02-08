from typing import Tuple, Optional
from torch import Tensor

from DeepPhysX_Core.Network.DataTransformation import DataTransformation


class TorchDataTransformation(DataTransformation):

    def __init__(self, config):
        """
        TorchDataTransformation is dedicated to data operations before and after network predictions.

        :param config: namedtuple containing the parameters of the network manager
        """
        super().__init__(config)
        self.data_type = Tensor

    @DataTransformation.check_type
    def transform_before_prediction(self, data_in: Tensor) -> Tensor:
        """
        Apply data operations before network's prediction.

        :param data_in: Input data
        :return: Transformed input data
        """
        return data_in

    @DataTransformation.check_type
    def transform_before_loss(self, data_out: Tensor, data_gt: Tensor = None) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Apply data operations between network's prediction and loss computation.

        :param data_out: Prediction data
        :param data_gt: Ground truth data
        :return: Transformed prediction data, transformed ground truth data
        """
        return data_out, data_gt

    @DataTransformation.check_type
    def transform_before_apply(self, data_out: Tensor) -> Tensor:
        """
        Apply data operations between loss computation and prediction apply in environment.

        :param data_out: Prediction data
        :return: Transformed prediction data
        """
        return data_out

    def __str__(self) -> str:
        description = "\n"
        description += f"  {self.__class__.__name__}\n"
        description += f"    Data type: {self.data_type}\n"
        description += f"    Transformation before prediction: Identity\n"
        description += f"    Transformation before loss: Identity\n"
        description += f"    Transformation before apply: Identity\n"
        return description
