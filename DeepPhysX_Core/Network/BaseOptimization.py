from typing import Dict, Union, Callable, Any

from dataclasses import dataclass
# from DeepPhysX_Core.Manager.Manager import Manager


class BaseOptimization:

    def __init__(self, config):
        """
        BaseOptimization is dedicated to network optimization: compute loss between prediction and target, update
        network parameters.

        :param config: namedtuple containing BaseOptimization parameters
        """

        self.manager: Any = None  # Manager
        # Loss
        self.loss_class = config.loss
        self.loss = None
        self.loss_value = 0.
        # Optimizer
        self.optimizer_class = config.optimizer
        self.optimizer = None
        self.lr = config.lr

    def set_loss(self) -> None:
        """Initialize the loss function"""
        raise NotImplementedError

    def compute_loss(self, prediction, ground_truth, data: Any) -> Dict[str, float]:
        """Compute loss from prediction / ground truth"""
        raise NotImplementedError

    def transform_loss(self, data: Any) -> Dict[str, float]:
        """Apply a transformation on the loss value using the potential additional data"""
        raise NotImplementedError

    def set_optimizer(self, net) -> None:
        """Define an optimization process"""
        raise NotImplementedError

    def optimize(self) -> None:
        """Run an optimization step"""
        raise NotImplementedError

    def __str__(self) -> str:
        """
        :return: String containing information about the BaseOptimization object
        """
        description = "\n"
        description += f"  {self.__class__.__name__}\n"
        description += f"    Loss class: {self.loss_class.__name__}\n" if self.loss_class else f"    Loss class: None\n"
        description += f"    Optimizer class: {self.optimizer_class.__name__}\n" if self.optimizer_class else \
            f"    Optimizer class: None\n"
        description += f"    Learning rate: {self.lr}\n"
        return description
