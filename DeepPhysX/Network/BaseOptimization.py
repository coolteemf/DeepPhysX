class BaseOptimization:

    def __init__(self, config):
        self.name = self.__class__.__name__
        # Loss
        self.loss_class = config.loss
        self.loss = None
        self.loss_value = 0.
        # Optimizer
        self.optimizer_class = config.optimizer
        self.optimizer = None
        self.lr = config.lr
        # Description
        self.description = ""

    def setLoss(self):
        raise NotImplementedError

    def computeLoss(self, prediction, ground_truth):
        raise NotImplementedError

    def setOptimizer(self, net):
        raise NotImplementedError

    def optimize(self):
        raise NotImplementedError

    def getDescription(self):
        if len(self.description) == 0:
            self.description += "\n{}\n".format(self.name)
            self.description += "   Loss class, loss: {}, {}\n".format(self.loss_class.__name__, self.loss)
            self.description += "   Learning rate: {}\n".format(self.lr)
            self.description += "   Optimizer class, optimizer: {}, {}\n".format(self.optimizer_class.__name__,
                                                                                 self.optimizer)
        return self.description
