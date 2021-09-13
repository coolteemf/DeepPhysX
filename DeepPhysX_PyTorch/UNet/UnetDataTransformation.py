import torch
import torch.nn.functional as F

from DeepPhysX_PyTorch.Network.TorchDataTransformation import TorchDataTransformation


class UnetDataTransformation(TorchDataTransformation):

    def __init__(self, network_config):
        super().__init__(network_config)

        self.grid_shape = self.network_config.network_config.grid_shape
        self.nb_classes = self.network_config.network_config.nb_classes
        self.nb_channels = self.network_config.network_config.nb_input_channels
        self.pad_widths = None
        self.data_scale = self.network_config.network_config.data_scale

    @TorchDataTransformation.check_type
    def transformBeforePrediction(self, data_in):
        data_in = torch.reshape(data_in,
                                (-1, self.grid_shape[2], self.grid_shape[1], self.grid_shape[0], self.nb_channels))
        data_in = data_in.permute((0, 4, 1, 2, 3))
        # Compute padding
        if self.pad_widths is None:
            element_shape = data_in[0].shape[1:]
            pad_widths, _ = self.network_config.in_out_pad_widths(element_shape)
            # torch applies padding from last dim to first
            pad_widths.reverse()
            pad = ()
            for p in pad_widths:
                pad += p
            self.pad_widths = pad + (0, 0, 0, 0)
        # Apply padding
        data_in = self.padding(data_in)
        return data_in

    @TorchDataTransformation.check_type
    def transformBeforeLoss(self, data_out, data_gt):
        # Transform ground truth
        data_gt = torch.reshape(data_gt,
                                (-1, self.grid_shape[2], self.grid_shape[1], self.grid_shape[0], self.nb_classes))
        data_gt = self.data_scale * data_gt
        # Transform prediction
        data_out = self.inverse_padding(data_out)
        data_out = data_out.permute(0, 2, 3, 4, 1)
        return data_out, data_gt

    @TorchDataTransformation.check_type
    def transformBeforeApply(self, data_out):
        data_out = data_out / self.data_scale
        return data_out

    def padding(self, data):
        return F.pad(data, self.pad_widths, mode='constant')

    def inverse_padding(self, data):
        pad_widths = []
        for i in range(len(self.pad_widths) // 2 - 1, -1, -1):
            pad_widths.append((self.pad_widths[2*i], self.pad_widths[2*i + 1]))
        def tuple_to_slice(v):
            if v is None:
                return slice(None)
            if v[1] == 0:
                return slice(v[0], None)
            return slice(v[0], -v[1])
        slices = tuple(map(tuple_to_slice, pad_widths))
        return data[slices]

    def __str__(self):
        description = "\n"
        description += f"  {self.name}\n"
        description += f"    Data type: {self.data_type}\n"
        description += f"    Data scale: {self.data_scale}\n"
        description += f"    Transformation before prediction: Input -> Reshape + Permute + Padding\n"
        description += f"    Transformation before loss: Ground Truth -> Reshape + Upscale / Prediction -> Inverse " \
                       f"padding + Permute\n"
        description += f"    Transformation before apply: Prediction -> Downscale\n"
        return description

