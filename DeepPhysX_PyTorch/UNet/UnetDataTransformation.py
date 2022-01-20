from typing import Any, List, Optional, Tuple

from click import Tuple
from torch.nn.functional import pad
from torch import reshape
from numpy import asarray

from DeepPhysX_PyTorch.Network.TorchDataTransformation import TorchDataTransformation, DataContainer


class UnetDataTransformation(TorchDataTransformation):

    def __init__(self, network_config):

        super().__init__(network_config)

        # Configure the transformation parameters
        self.input_size: List[int] = self.network_config.network_config.input_size
        self.nb_steps: int = self.network_config.network_config.nb_steps
        self.nb_output_channels: int = self.network_config.network_config.nb_output_channels
        self.nb_input_channels: int = self.network_config.network_config.nb_input_channels
        self.data_scale: float = self.network_config.network_config.data_scale
        self.pad_widths: Optional[List[int]] = None
        self.inverse_pad_widths: Optional[List[int]] = None

        # Define shape transformations
        border = 4 if self.network_config.network_config.two_sublayers else 2
        border = 0 if self.network_config.network_config.border_mode == 'same' else border
        self.reverse_first_step = lambda x: x + border
        self.reverse_down_step = lambda x: (x + border) * 2
        self.reverse_up_step = lambda x: (x + border - 1) // 2 + 1

    @TorchDataTransformation.check_type
    def transformBeforePrediction(self, data_in: DataContainer) -> DataContainer:
        """
        Apply data operations before network's prediction.

        :param data_in: Input data
        :return: Transformed input data
        """
        # Transform tensor shape
        data_in = data_in.view((-1, self.input_size[2], self.input_size[1], self.input_size[0], self.nb_input_channels))
        data_in = data_in.permute((0, 4, 1, 2, 3))

        # Compute padding
        if self.pad_widths is None:
            transformed_shape = data_in[0].shape[1:]
            self.compute_pad_widths(transformed_shape)

        # Apply padding
        data_in = pad(data_in, self.pad_widths, mode='constant')
        return data_in

    @TorchDataTransformation.check_type
    def transformBeforeLoss(self, data_out: DataContainer, data_gt: DataContainer) -> Tuple[DataContainer, DataContainer]:
        """
        Apply data operations between network's prediction and loss computation.

        :param data_out: Prediction data
        :param data_gt: Ground truth data
        :return: Transformed prediction data, transformed ground truth data
        """
        # Transform ground truth
        # Transform tensor shape, apply scale
        if data_gt is not None:
            data_gt = reshape(data_gt,
                              (-1, self.input_size[2], self.input_size[1], self.input_size[0], self.nb_output_channels))
            data_gt = self.data_scale * data_gt

        # Transform prediction
        # Apply inverse padding, permute
        data_out = pad(data_out, self.inverse_pad_widths)
        data_out = data_out.permute(0, 2, 3, 4, 1)

        return data_out, data_gt

    @TorchDataTransformation.check_type
    def transformBeforeApply(self, data_out: DataContainer) -> DataContainer:
        """
        Apply data operations between loss computation and prediction apply in environment.

        :param data_out: Prediction data
        :return: Transformed prediction data
        """
        # Rescale prediction
        data_out = data_out / self.data_scale
        return data_out

    def compute_pad_widths(self, desired_shape: List[int]) -> None:

        # Compute minimal input shape given the desired shape
        minimal_shape = asarray(desired_shape)
        for i in range(self.nb_steps):
            minimal_shape = self.reverse_up_step(minimal_shape)
        for i in range(self.nb_steps):
            minimal_shape = self.reverse_down_step(minimal_shape)
        minimal_shape = tuple(self.reverse_first_step(minimal_shape))

        # Compute padding width between shapes
        pad_widths = [((m - d) // 2, (m - d - 1) // 2 + 1) for m, d in zip(minimal_shape, desired_shape)]
        pad_widths.reverse()    # PyTorch applies padding from last dimension
        self.pad_widths, self.inverse_pad_widths = (), ()
        for p in pad_widths:
            self.pad_widths += p
            self.inverse_pad_widths += (-p[0], -p[1])   # PyTorch accepts negative padding

    def __str__(self):
        description = "\n"
        description += f"  {self.name}\n"
        description += f"    Data type: {self.data_type}\n"
        description += f"    Data scale: {self.data_scale}\n"
        description += f"    Transformation before prediction: Input -> Reshape + Permute + Padding\n"
        description += f"    Transformation before loss: Ground Truth -> Reshape + Upscale\n"
        description += f"                                Prediction -> Inverse padding + Permute\n"
        description += f"    Transformation before apply: Prediction -> Downscale\n"
        return description
