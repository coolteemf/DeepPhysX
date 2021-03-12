from .Network import Network
from .NetworkOptimization import NetworkOptimization


class NetworkConfig:

    def __init__(self, network_name="", network_type="", network_dir=None, loss=None, lr=None, optimizer=None,
                 save_each_epoch=False, which_network=1):

        self.network = Network
        self.networkConfig = network_name, network_type

        self.optimization = NetworkOptimization
        self.optimizationConfig = loss, lr, optimizer
        self.trainingMaterials = (lr is not None) and (optimizer is not None)

        self.networkDir = network_dir
        self.existingNetwork = False if network_dir is None else True
        self.whichNetwork = which_network

        self.saveEachEpoch = save_each_epoch and self.trainingMaterials

        self.description = ""

    def createNetwork(self):
        return self.network(*self.networkConfig)

    def createOptimization(self):
        return self.optimization(*self.optimizationConfig)

    def getDescription(self):
        if len(self.description) == 0:
            self.description += "\nNETWORK CONFIG\n"
            self.description += "   Network class: {}\n".format(self.network)
            self.description += "   Config: {}\n".format(self.networkConfig)
        return self.description
