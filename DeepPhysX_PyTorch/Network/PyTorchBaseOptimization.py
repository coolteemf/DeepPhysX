import torch
import numpy as np

from DeepPhysX.Network.BaseOptimization import BaseOptimization


class PyTorchBaseOptimization(BaseOptimization):

    def __init__(self, config):
        BaseOptimization.__init__(self, config)
        self.descriptionName = "PYTORCH NetworkOptimization"

    def setLoss(self):
        if self.loss_class is not None:
            self.loss = self.loss_class()

    def computeLoss(self, prediction, ground_truth):
        prediction = torch.from_numpy(prediction) if type(prediction) is np.ndarray else prediction
        ground_truth = torch.from_numpy(ground_truth) if type(ground_truth) is np.ndarray else ground_truth
        return self.loss(prediction, ground_truth)

    def setOptimizer(self, net):
        if (self.optimizer_class is not None) and (self.lr is not None):
            self.optimizer = self.optimizer_class(net.parameters(), self.lr)

    def optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
