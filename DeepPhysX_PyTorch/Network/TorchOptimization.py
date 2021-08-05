from DeepPhysX_Core.Network.BaseOptimization import BaseOptimization


class TorchOptimization(BaseOptimization):

    def __init__(self, config):
        BaseOptimization.__init__(self, config)

    def setLoss(self):
        if self.loss_class is not None:
            self.loss = self.loss_class()

    def computeLoss(self, prediction, ground_truth):
        self.loss_value = self.loss(prediction, ground_truth)
        return self.loss_value.item()

    def setOptimizer(self, net):
        if (self.optimizer_class is not None) and (self.lr is not None):
            self.optimizer = self.optimizer_class(net.parameters(), self.lr)

    def optimize(self):
        self.optimizer.zero_grad()
        self.loss_value.backward()
        self.optimizer.step()

    def __str__(self):
        description = BaseOptimization.__str__(self)
        return description
