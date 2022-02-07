from DeepPhysX_Core.Network.BaseNetwork import BaseNetwork
from DeepPhysX_Core.Network.BaseOptimization import BaseOptimization


class DummyNetwork(BaseNetwork):

    def __init__(self, config):
        BaseNetwork.__init__(self, config)

    def forward(self, x):
        return x

    def set_train(self):
        pass

    def set_eval(self):
        pass

    def set_device(self):
        pass

    def load_parameters(self, path):
        pass

    def get_parameters(self):
        pass

    def save_parameters(self, path):
        pass

    def nb_parameters(self):
        return 0

    def transform_from_numpy(self, x, grad=True):
        return x

    def transform_to_numpy(self, x):
        return x


class DummyOptimizer(BaseOptimization):

    def __init__(self, config):
        BaseOptimization.__init__(self, config)

    def set_loss(self):
        pass

    def compute_loss(self, prediction, ground_truth, data):
        return {'loss': 0.}

    def transform_loss(self, data):
        pass

    def set_optimizer(self, net):
        pass

    def optimize(self):
        pass
