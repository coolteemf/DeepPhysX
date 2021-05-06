from DeepPhysX.Network.BaseOptimization import BaseOptimization


class PyTorchBaseOptimization(BaseOptimization):

    def __init__(self, config):
        BaseOptimization.__init__(self, config)
        self.descriptionName = "PYTORCH NetworkOptimization"

    def setLoss(self):
        if self.loss_class is not None:
            self.loss = self.loss_class()

    def computeLoss(self, prediction, ground_truth):
        self.loss_value = self.loss(prediction, ground_truth)
        return self.loss_value

    def setOptimizer(self, net):
        if (self.optimizer_class is not None) and (self.lr is not None):
            self.optimizer = self.optimizer_class(net.parameters(), self.lr)

    def optimize(self):
        self.optimizer.zero_grad()
        self.loss_value.backward()
        self.optimizer.step()
