class BaseOptimization:

    def __init__(self, config):
        self.loss_class = config.loss
        self.loss = None
        self.lr = config.lr
        self.optimizer_class = config.optimizer
        self.optimizer = None
        # Description
        self.descriptionName = "CORE NetworkOptimization"
        self.description = ""

    def setLoss(self):
        # if loss is not None:
        #       self.loss = self.loss_class()
        raise NotImplementedError

    def computeLoss(self, prediction, ground_truth):
        # return self.loss(prediction, ground_truth)
        raise NotImplementedError

    def setOptimizer(self, net):
        # if self.optimizer_class is not None and lr is not None:
        #       self.optimizer = self.optimizer_class(net.parameters(), self.lr)
        raise NotImplementedError

    def optimize(self, loss):
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        raise NotImplementedError

    def getDescription(self):
        if len(self.description) == 0:
            self.description += "\n{}\n".format(self.descriptionName)
            self.description += "   Loss class, loss: {}, {}\n".format(self.loss_class.__name__, self.loss)
            self.description += "   Learning rate: {}\n".format(self.lr)
            self.description += "   Optimizer class, optimizer: {}, {}\n".format(self.optimizer_class.__name__,
                                                                                 self.optimizer)
        return self.description
