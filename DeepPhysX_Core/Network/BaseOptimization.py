from typing import Optional, Dict, Union, Callable, Any

from dataclasses import dataclass
# from DeepPhysX_Core.Manager.Manager import Manager


class BaseOptimization:

    @dataclass
    class BaseOptimizationProperties:
        """
        Class containing data to create BaseOptimization objects.
        """
        loss: Union[Callable[[Any, Any], Any], Callable[[Any], Any]]
        lr: float
        optimizer: Any

    def __init__(self, config: BaseOptimizationProperties):
        """
        BaseOptimization is dedicated to network optimization: compute loss between prediction and target, update
        network parameters.

        :param BaseNetworkConfig.BaseOptimizationProperties config: BaseOptimizationProperties class containing BaseOptimization parameters
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
        raise NotImplementedError

    def compute_loss(self, prediction, ground_truth, data) -> Dict[str, float]:
        raise NotImplementedError

    def transform_loss(self, data) -> Dict[str, float]:
        raise NotImplementedError

    def set_optimizer(self, net) -> None:
        raise NotImplementedError

    def optimize(self) -> None:
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
