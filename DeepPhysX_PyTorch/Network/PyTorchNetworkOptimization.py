import torch

from DeepPhysX.Network.NetworkOptimization import NetworkOptimization


class PyTorchNetworkOptimization(NetworkOptimization):

    def __init__(self, loss, lr, optimizer):
        NetworkOptimization.__init__(self, loss, lr, optimizer)

    def setLoss(self):
        if self.loss_class is not None:
            self.loss = self.loss_class()

    def computeLoss(self, prediction, ground_truth):
        ground_truth = torch.from_numpy(ground_truth)
        return self.loss(prediction, ground_truth)

    def setOptimizer(self, net):
        if (self.optimizer_class is not None) and (self.lr is not None):
            self.optimizer = self.optimizer_class(net.parameters(), self.lr)

    def optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
