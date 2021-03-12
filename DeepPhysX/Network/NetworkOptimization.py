class NetworkOptimization:

    def __init__(self, loss, lr, optimizer):
        self.loss = loss
        self.lr = lr
        self.optimizer = optimizer

    def setLoss(self):
        # Todo: if loss is not None
        # return self.loss()
        raise NotImplementedError

    def setOptimizer(self):
        # Todo: if optimizer and lr are not None
        # return self.optimizer(self.lr), self.lr
        raise NotImplementedError
