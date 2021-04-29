import numpy as np

from DeepPhysX.Network.BaseNetwork import BaseNetwork
from DeepPhysX.Network.BaseOptimization import BaseOptimization


class MyBaseNetwork(BaseNetwork):

    def __init__(self, network_name="", *args):
        BaseNetwork.__init__(self, network_name=network_name, network_type="MyNetwork")
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
        params = np.loadtxt(path)
        self.a, self.b, self.c, self.d = params[0], params[1], params[2], params[3]

    def getParameters(self):
        return np.array([self.a, self.b, self.c, self.d])

    def saveParameters(self, path):
        np.savetxt(path, self.getParameters())

    def nbParameters(self):
        return len(self.getParameters())

    def transformFromNumpy(self, x):
        return x

    def transformToNumpy(self, x):
        return x


class MyBaseOptimisation(BaseOptimization):

    def __init__(self, loss, lr, optimizer):
        BaseOptimization.__init__(self, loss=None, lr=lr, optimizer=None)
        self.net = None

    def setLoss(self):
        pass

    def computeLoss(self, prediction, ground_truth):
        return {'item': np.square(prediction - ground_truth).sum(),
                'grad': 2.0 * (prediction - ground_truth)}

    def setOptimizer(self, net):
        self.net = net

    def optimize(self, loss):
        grad = loss['grad']
        data = self.net.data
        grad_a = grad.sum()
        grad_b = (grad * data).sum()
        grad_c = (grad * data ** 2).sum()
        grad_d = (grad * data ** 3).sum()
        self.net.a -= self.lr * grad_a
        self.net.b -= self.lr * grad_b
        self.net.c -= self.lr * grad_c
        self.net.d -= self.lr * grad_d
