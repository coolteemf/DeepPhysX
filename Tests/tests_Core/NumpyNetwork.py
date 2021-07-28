import numpy as np
from dataclasses import dataclass

from DeepPhysX_Core.Network.BaseNetworkConfig import BaseNetworkConfig
from DeepPhysX_Core.Network.BaseNetwork import BaseNetwork
from DeepPhysX_Core.Network.BaseOptimization import BaseOptimization
from DeepPhysX_Core.Network.DataTransformation import DataTransformation


class NumpyNetwork(BaseNetwork):

    def __init__(self, config):
        BaseNetwork.__init__(self, config)
        self.data = None
        self.p = [np.random.randn() for _ in range(self.config.nb_parameters)]

    def forward(self, x):
        self.data = x
        output = 0
        for i in range(len(self.p)):
            output += self.p[i] * (x ** i)
        return output

    def setTrain(self):
        pass

    def setEval(self):
        pass

    def setDevice(self):
        pass

    def loadParameters(self, path):
        self.p = np.load(path)

    def getParameters(self):
        return np.array(self.p)

    def saveParameters(self, path):
        np.save(path, self.getParameters())

    def nbParameters(self):
        return len(self.p)

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
        for i in range(self.net.nbParameters()):
            grad_p_i = (grad * (data ** i)).sum()
            self.net.p[i] -= self.lr * grad_p_i


class NumpyTransformation(DataTransformation):

    def __init__(self, network_config):
        super(NumpyTransformation, self).__init__(network_config)
        self.data_type = np.ndarray

    @DataTransformation.check_type
    def transformBeforePrediction(self, data_in):
        return data_in

    @DataTransformation.check_type
    def transformBeforeLoss(self, data_out, data_gt):
        return data_out, data_gt

    @DataTransformation.check_type
    def transformBeforeApply(self, data_out):
        return data_out


class NumpyNetworkConfig(BaseNetworkConfig):

    @dataclass
    class NumpyNetworkProperties(BaseNetworkConfig.BaseNetworkProperties):
        nb_parameters: int

    def __init__(self, network_class=NumpyNetwork, optimization_class=NumpyOptimisation,
                 data_transformation_class=NumpyTransformation, network_dir=None, network_name='NumpyNetwork',
                 network_type='Regression', which_network=0, save_each_epoch=True, lr=None, nb_parameters=4):
        super(NumpyNetworkConfig, self).__init__(network_class=network_class, optimization_class=optimization_class,
                                                 data_transformation_class=data_transformation_class,
                                                 network_dir=network_dir, which_network=which_network,
                                                 save_each_epoch=save_each_epoch, lr=lr)
        self.network_config = self.NumpyNetworkProperties(network_name=network_name, network_type=network_type,
                                                          nb_parameters=nb_parameters)
