from .NetworkOptimization import NetworkOptimization


class NetworkConfig:

    def __init__(self, network_class, network_name="", network_type="", loss=None, lr=None, optimizer=None,
                 network_dir=None, save_each_epoch=False, which_network=None):
        # Network variables
        self.network_class = network_class
        self.networkConfig = network_name, network_type
        # Optimization variables
        self.optimization_class = NetworkOptimization
        self.optimizationConfig = loss, lr, optimizer
        self.trainingMaterials = (lr is not None) and (optimizer is not None)
        # NetworkManager variables
        self.networkDir = network_dir
        self.existingNetwork = False if network_dir is None else True
        self.whichNetwork = which_network
        self.saveEachEpoch = save_each_epoch and self.trainingMaterials
        # Description
        self.descriptionName = "CORE NetworkConfig"
        self.description = ""

    def createNetwork(self):
        return self.network_class(*self.networkConfig)

    def createOptimization(self):
        return self.optimization_class(*self.optimizationConfig)

    def getDescription(self):
        if len(self.description) == 0:
            self.description += "\n{}\n".format(self.descriptionName)
            self.description += "   (network) Network class: {}\n".format(self.network_class.__name__)
            self.description += "   (network) Network config: {}\n".format(self.networkConfig)
            self.description += "   (optimization) Optimization class: {}\n".format(self.optimization_class.__name__)
            self.description += "   (optimization) Optimization config: {}\n".format(self.optimizationConfig)
            self.description += "   (optimization) Training materials: {}\n".format(self.trainingMaterials)
            self.description += "   (networkManager) Network directory: {}\n".format(self.networkDir)
            self.description += "   (networkManager) Existing network: {}\n".format(self.existingNetwork)
            self.description += "   (networkManager) Which network: {}\n".format(self.whichNetwork)
            self.description += "   (networkManager) Save each epoch: {}\n".format(self.saveEachEpoch)
        return self.description
