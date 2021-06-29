from DeepPhysX.Network.BaseOptimization import BaseOptimization


class TorchOptimization(BaseOptimization):

    def __init__(self, config):
        BaseOptimization.__init__(self, config)

    def setLoss(self):
        if self.loss_class is not None:
            self.loss = self.loss_class()

    def computeLoss(self, prediction, ground_truth):
        self.p, self.gt = prediction, ground_truth
        # print("prediction\n", prediction)
        # print("ground truth\n", ground_truth)
        self.loss_value = self.loss(prediction, ground_truth)
        return self.loss_value.item()

    def setOptimizer(self, net):
        if (self.optimizer_class is not None) and (self.lr is not None):
            self.optimizer = self.optimizer_class(net.parameters(), self.lr)

    def optimize(self):
        self.optimizer.zero_grad()
        self.p.retain_grad()
        self.loss_value.backward()
        self.optimizer.step()