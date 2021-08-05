class BaseOptimization:

    def __init__(self, config):
        """
        BaseOptimization is dedicated to network optimization: compute loss between prediction and target, update
        network parameters.

        :param config: BaseNetworkConfig.BaseOptimizationProperties class containing BaseOptimization parameters
        """
        self.name = self.__class__.__name__
        # Loss
        self.loss_class = config.loss
        self.loss = None
        self.loss_value = 0.
        # Optimizer
        self.optimizer_class = config.optimizer
        self.optimizer = None
        self.lr = config.lr

    def setLoss(self):
        raise NotImplementedError

    def computeLoss(self, prediction, ground_truth):
        raise NotImplementedError

    def setOptimizer(self, net):
        raise NotImplementedError

    def optimize(self):
        raise NotImplementedError

    def __str__(self):
        """
        :return: String containing information about the BaseOptimization object
        """
        description = "\n"
        description += f"  {self.name}\n"
        description += f"    Loss class: {self.loss_class.__name__}\n"
        description += f"    Optimizer class: {self.optimizer_class.__name__}\n"
        description += f"    Learning rate: {self.lr}\n"
        return description
