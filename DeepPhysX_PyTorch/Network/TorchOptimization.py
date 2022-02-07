from typing import Dict

from DeepPhysX_Core.Network.BaseOptimization import BaseOptimization


class TorchOptimization(BaseOptimization):

    def __init__(self, config):
        """
        TorchOptimization is dedicated to network optimization: compute loss between prediction and target, update
        network parameters.

        :param config: namedtuple containing TorchOptimization parameters
        """
        BaseOptimization.__init__(self, config)

    def set_loss(self) -> None:
        """Initialize the loss function"""
        if self.loss_class is not None:
            self.loss = self.loss_class()

    def compute_loss(self, prediction, ground_truth, data) -> Dict[str, float]:
        """Compute loss from prediction / ground truth"""
        self.loss_value = self.loss(prediction, ground_truth)
        return self.transform_loss(data)

    def transform_loss(self, data) -> Dict[str, float]:
        """Apply a transformation on the loss value using the potential additional data"""
        return {'loss': self.loss_value.item()}

    def set_optimizer(self, net) -> None:
        """Define an optimization process"""
        if (self.optimizer_class is not None) and (self.lr is not None):
            self.optimizer = self.optimizer_class(net.parameters(), self.lr)

    def optimize(self) -> None:
        """Run an optimization step"""
        self.optimizer.zero_grad()
        self.loss_value.backward()
        self.optimizer.step()

    def __str__(self) -> str:
        return BaseOptimization.__str__(self)
