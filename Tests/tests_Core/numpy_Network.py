import numpy as np

from DeepPhysX.Network.BaseNetwork import BaseNetwork
from DeepPhysX.Network.BaseOptimization import BaseOptimization


class NumpyNetwork(BaseNetwork):

    def __init__(self, config):
        BaseNetwork.__init__(self, config)
        self.a = np.random.randn()
        self.b = np.random.randn()
        self.c = np.random.randn()
        self.d = np.random.randn()
        self.data = None

    def forward(self, x):
        self.data = x
        return self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3

    def setTrain(self):
        pass

    def setEval(self):
        pass

    def setDevice(self):
        pass

    def loadParameters(self, path):
        params = np.load(path)
        self.a, self.b, self.c, self.d = params[0], params[1], params[2], params[3]

    def getParameters(self):
        return np.array([self.a, self.b, self.c, self.d])

    def saveParameters(self, path):
        np.save(path, self.getParameters())

    def nbParameters(self):
        return len(self.getParameters())

    def transformFromNumpy(self, x):
        return x

    def transformToNumpy(self, x):
        return x


class NumpyOptimisation(BaseOptimization):

    def __init__(self, config):
        BaseOptimization.__init__(self, config)
        self.net = None

    def setLoss(self):
        pass

    def computeLoss(self, prediction, ground_truth):
        self.loss_value = {'item': np.square(prediction - ground_truth).sum(),
                           'grad': 2.0 * (prediction - ground_truth)}
        return self.loss_value['item']

    def setOptimizer(self, net):
        self.net = net

    def optimize(self):
        grad = self.loss_value['grad']
        data = self.net.data
        grad_a = grad.sum()
        self.net.a -= self.lr * grad_a
        grad_b = (grad * data).sum()
        self.net.b -= self.lr * grad_b
        grad_c = (grad * data ** 2).sum()
        self.net.c -= self.lr * grad_c
        grad_d = (grad * data ** 3).sum()
        self.net.d -= self.lr * grad_d
