from torch import Tensor

from DeepPhysX_Core.Network.DataTransformation import DataTransformation


class TorchDataTransformation(DataTransformation):

    def __init__(self, network_config):
        super().__init__(network_config)
        self.data_type = Tensor

    @DataTransformation.check_type
    def transformBeforePrediction(self, data_in):
        return data_in

    @DataTransformation.check_type
    def transformBeforeLoss(self, data_out, data_gt):
        return data_out, data_gt

    @DataTransformation.check_type
    def transformBeforeApply(self, data_out):
        return data_out

    def __str__(self):
        description = "\n"
        description += f"  {self.name}\n"
        description += f"    Data type: {self.data_type}\n"
        description += f"    Transformation before prediction: Identity\n"
        description += f"    Transformation before loss: Identity\n"
        description += f"    Transformation before apply: Identity\n"
        return description
