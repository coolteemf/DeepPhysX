from typing import Dict

from DeepPhysX_Core.Network.BaseOptimization import BaseOptimization


class TorchOptimization(BaseOptimization):

    def __init__(self, config):
        BaseOptimization.__init__(self, config)

    def set_loss(self) -> None:
        if self.loss_class is not None:
            self.loss = self.loss_class()

    def compute_loss(self, prediction, ground_truth, data) -> Dict[str, float]:
        self.loss_value = self.loss(prediction, ground_truth)
        return self.transform_loss(data)

    def transform_loss(self, data) -> Dict[str, float]:
        return {'loss': self.loss_value.item()}

    def set_optimizer(self, net) -> None:
        if (self.optimizer_class is not None) and (self.lr is not None):
            self.optimizer = self.optimizer_class(net.parameters(), self.lr)

    def optimize(self) -> None:
        self.optimizer.zero_grad()
        self.loss_value.backward()
        self.optimizer.step()

    def __str__(self) -> str:
        return BaseOptimization.__str__(self)
