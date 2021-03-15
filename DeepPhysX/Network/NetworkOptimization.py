class NetworkOptimization:

    def __init__(self, loss, lr, optimizer):
        self.loss_class = loss
        self.loss = None
        self.lr = lr
        self.optimizer_class = optimizer
        self.optimizer = None

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
