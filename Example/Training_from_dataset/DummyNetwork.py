from DeepPhysX_Core.Network.BaseNetwork import BaseNetwork
from DeepPhysX_Core.Network.BaseOptimization import BaseOptimization


class DummyNetwork(BaseNetwork):

    def __init__(self, config):
        BaseNetwork.__init__(self, config)

    def forward(self, x):
        return x

    def setTrain(self):
        pass

    def setEval(self):
        pass

    def setDevice(self):
        pass

    def loadParameters(self, path):
        pass

    def getParameters(self):
        pass

    def saveParameters(self, path):
        pass

    def nbParameters(self):
        return 0

    def transformFromNumpy(self, x, grad=True):
        return x

    def transformToNumpy(self, x):
        return x


class DummyOptimizer(BaseOptimization):

    def __init__(self, config):
        BaseOptimization.__init__(self, config)

    def setLoss(self):
        pass

    def computeLoss(self, prediction, ground_truth, data):
        print(prediction, ground_truth)
        return {'loss': 0.}

    def transformLoss(self, data):
        pass

    def setOptimizer(self, net):
        pass

    def optimize(self):
        pass
