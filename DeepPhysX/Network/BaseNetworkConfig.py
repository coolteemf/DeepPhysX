from dataclasses import dataclass
from typing import Any

from DeepPhysX.Network.BaseNetwork import BaseNetwork
from DeepPhysX.Network.BaseOptimization import BaseOptimization


class BaseNetworkConfig:

    @dataclass
    class BaseNetworkProperties:
        network_name: str
        network_type: str

    @dataclass
    class BaseOptimizationProperties:
        loss: Any
        lr: float
        optimizer: Any

    def __init__(self, network_class=BaseNetwork, optimization_class=BaseOptimization, network_dir=None,
                 network_name='Network', network_type='BaseNetwork', which_network=0, save_each_epoch=False,
                 loss=None, lr=None, optimizer=None):

        self.name = self.__class__.__name__

        # Check the arguments before to configure anything
        if network_dir is not None and type(network_dir) != str:
            raise TypeError("[{}] The network directory must be a str.".format(self.name))
        if type(network_name) != str:
            raise TypeError("[{}] The network name must be a str.".format(self.name))
        if type(network_type) != str:
            raise TypeError("[{}] The network type must be a str.".format(self.name))
        if type(which_network) != int:
            raise TypeError("[{}] The argument 'which network' must be an int.".format(self.name))
        if type(save_each_epoch) != bool:
            raise TypeError("[{}] The argument 'save each epoch' must be set to True or False.".format(self.name))

        # Network class
        self.network_class = network_class
        # Network configuration
        self.network_config = self.BaseNetworkProperties(network_name=network_name, network_type=network_type)

        # Optimization class
        self.optimization_class = optimization_class
        # Optimization configuration
        self.optimization_config = self.BaseOptimizationProperties(loss=loss, lr=lr, optimizer=optimizer)
        self.training_stuff = (loss is not None) and (optimizer is not None)

        # NetworkManager variables
        self.network_dir = network_dir
        self.existing_network = False if network_dir is None else True
        self.which_network = which_network
        self.save_each_epoch = save_each_epoch and self.training_stuff

        # Description
        self.description = ""

    def createNetwork(self):
        try:
            network = self.network_class(self.network_config)
        except:
            raise TypeError("[{}] The given network class is not a BaseNetwork child class.".format(self.name))
        if not isinstance(network, BaseNetwork):
            raise TypeError("[{}] The network class must be a BaseNetwork child object.".format(self.name))
        return network

    def createOptimization(self):
        try:
            optimization = self.optimization_class(self.optimization_config)
        except:
            raise TypeError("[{}] The given optimization class is not a BaseOptimization child class.".format(self.name))
        if not isinstance(optimization, BaseOptimization):
            raise TypeError("[{}] The optimization class must be a BaseOptimization child object.".format(self.name))
        return optimization

    def getDescription(self):
        if len(self.description) == 0:
            self.description += "\n{}\n".format(self.name)
            self.description += "   (network) Network class: {}\n".format(self.network_class.__name__)
            self.description += "   (network) Network config: {}\n".format(self.network_config)
            self.description += "   (optimization) Optimization class: {}\n".format(self.optimization_class.__name__)
            self.description += "   (optimization) Optimization config: {}\n".format(self.optimization_config)
            self.description += "   (optimization) Training materials: {}\n".format(self.training_stuff)
            self.description += "   (networkManager) Network directory: {}\n".format(self.network_dir)
            self.description += "   (networkManager) Existing network: {}\n".format(self.existing_network)
            self.description += "   (networkManager) Which network: {}\n".format(self.which_network)
            self.description += "   (networkManager) Save each epoch: {}\n".format(self.save_each_epoch)
        return self.description
