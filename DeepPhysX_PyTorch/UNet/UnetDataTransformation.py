import numpy as np

from DeepPhysX_Core.Network.DataTransformation import DataTransformation


class UnetDataTransformation(DataTransformation):

    def __init__(self, network_config):
        super().__init__(network_config)

        self.grid_shape = self.network_config.network_config.grid_shape
        self.nb_classes = self.network_config.network_config.nb_classes
        self.nb_channels = self.network_config.network_config.nb_input_channels
        self.pad_widths = None

    def transformBeforePrediction(self, x, gt):
        x = np.reshape(x, (-1, self.grid_shape[2], self.grid_shape[1], self.grid_shape[0], self.nb_channels))
        x = np.transpose(x, (0, 4, 1, 2, 3))
        gt = np.reshape(gt, (-1, self.grid_shape[2], self.grid_shape[1], self.grid_shape[0], self.nb_classes))
        gt = np.transpose(gt, (0, 4, 1, 2, 3))
        # Compute padding
        if self.pad_widths is None:
            element_shape = x[0].shape[1:]
            pad_widths, _ = self.network_config.in_out_pad_widths(element_shape)
            self.pad_widths = [(0, 0), (0, 0)] + pad_widths
        # Apply padding
        x = self.padding(x)
        gt = self.padding(gt)
        return x, gt

    def transformAfterPrediction(self, y, gt):
        # Inverse padding
        y = self.inverse_padding(y)
        gt = self.inverse_padding(gt)
        # y = np.transpose(y, (0, 2, 3, 4, 1))
        # gt = np.transpose(gt, (0, 2, 3, 4, 1))
        y.permute(0, 2, 3, 4, 1)
        gt.permute(0, 2, 3, 4, 1)
        return y, gt

    def padding(self, data):
        return np.pad(data, self.pad_widths, mode='constant')

    def inverse_padding(self, data):
        def tuple_to_slice(v):
            if v is None:
                return slice(None)
            if v[1] == 0:
                return slice(v[0], None)
            return slice(v[0], -v[1])
        slices = tuple(map(tuple_to_slice, self.pad_widths))
        return data[slices]
