import torch
import numpy as np

from DeepPhysX_PyTorch.Network.TorchOptimization import TorchOptimization


class PhysicBasedOptimization(TorchOptimization):

    def __init__(self, config):
        TorchOptimization.__init__(self, config)

    def transformLoss(self, data):
        residual_loss = torch.tensor(3000, dtype=torch.float, device=data.device) if torch.isnan(
             data).any() else data.mean()
        self.loss_value.retain_grad()
        mse_loss = self.loss_value.item()
        # self.loss_value = self.loss_value * residual_loss
        return {'loss': self.loss_value.item(),
                'MSE_loss': mse_loss,
                'residual_loss': residual_loss.item()}

    # def transformLoss(self, data):
    #     return {'loss': self.loss_value.item()}
