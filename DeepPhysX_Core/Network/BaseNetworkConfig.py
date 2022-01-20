from dataclasses import dataclass
from typing import Any, Optional, Union, Callable
import numpy
import torch
from os.path import isdir

from DeepPhysX_Core.Network.BaseNetwork import BaseNetwork
from DeepPhysX_Core.Network.BaseOptimization import BaseOptimization
from DeepPhysX_Core.Network.DataTransformation import DataTransformation

DataContainer = Union[numpy.ndarray, torch.Tensor]


class BaseNetworkConfig:
    @dataclass
    class BaseNetworkProperties:
        """
        Class containing data to create BaseNetwork objects.
        """
        network_name: str
        network_type: str

    def __init__(self,
                 network_class: BaseNetwork,
                 optimization_class: BaseOptimization,
                 data_transformation_class: DataTransformation,
                 network_dir: str = None,
                 network_name: str = 'Network',
                 network_type: str = 'BaseNetwork',
                 which_network: int = 0,
                 save_each_epoch: bool = False,
                 loss: Any = None,
                 lr: Optional[float] = None,
                 optimizer: Any = None):
        """
        BaseNetworkConfig is a configuration class to parameterize and create BaseNetwork, BaseOptimization and
        DataTransformation for the NetworkManager.

        :param network_class: BaseNetwork class from which an instance will be created
        :type network_class: BaseNetwork
        :param optimization_class: BaseOptimization class from which an instance will be created
        :type optimization_class: BaseOptimization
        :param data_transformation_class: DataTransformation class from which an instance will be created
        :type data_transformation_class: DataTransformation
        :param str network_dir: Name of an existing network repository
        :param str network_name: Name of the network
        :param str network_type: Type of the network
        :param int which_network: If several networks in network_dir, load the specified one
        :param bool save_each_epoch: If True, network state will be saved at each epoch end; if False, network state
                                     will be saved at the end of the training
        :param loss: Loss class
        :param float lr: Learning rate
        :param optimizer: Network's parameters optimizer class
        """

        # Check network_dir type and existence
        if network_dir is not None:
            if type(network_dir) != str:
                raise TypeError(
                    f"[{self.__class__.__name__}] Wrong 'network_dir' type: str required, get{type(network_dir)}")
            if not isdir(network_dir):
                raise ValueError(f"[{self.__class__.__name__}] Given 'network_dir' does not exists: {network_dir}")
        # Check network_name type
        if type(network_name) != str:
            raise TypeError(
                f"[{self.__class__.__name__}] Wrong 'network_name' type: str required, get {type(network_name)}")
        # Check network_tpe type
        if type(network_type) != str:
            raise TypeError(
                f"[{self.__class__.__name__}] Wrong 'network_type' type: str required, get {type(network_type)}")
        # Check which_network type and value
        if type(which_network) != int:
            raise TypeError(
                f"[{self.__class__.__name__}] Wrong 'which_network' type: int required, get {type(which_network)}")
        if which_network < 0:
            raise ValueError(f"[{self.__class__.__name__}] Given 'which_network' value is negative")
        # Check save_each_epoch type
        if type(save_each_epoch) != bool:
            raise TypeError(
                f"[{self.__class__.__name__}] Wrong 'save each epoch' type: bool required, get {type(save_each_epoch)}")

        # BaseNetwork parameterization
        self.network_class: BaseNetwork = network_class
        self.network_config: Any = self.BaseNetworkProperties(network_name=network_name, network_type=network_type)

        # BaseOptimization parameterization
        self.optimization_class: Any = optimization_class
        self.optimization_config: Any = BaseOptimization.BaseOptimizationProperties(loss=loss, lr=lr, optimizer=optimizer)
        self.training_stuff: bool = (loss is not None) and (optimizer is not None)

        # NetworkManager parameterization
        self.data_transformation_class: Any = data_transformation_class
        self.network_dir: str = network_dir
        self.which_network: int = which_network
        self.save_each_epoch: bool = save_each_epoch and self.training_stuff

    def create_network(self) -> BaseNetwork:
        """
        :return: BaseNetwork object from network_class and its parameters.
        """
        try:
            network = self.network_class(config=self.network_config)
        except:
            raise ValueError(
                f"[{self.__class__.__name__}] Given 'network_class' cannot be created in {self.__class__.__name__}")
        if not isinstance(network, BaseNetwork):
            raise TypeError(
                f"[{self.__class__.__name__}] Wrong 'network_class' type: BaseNetwork required, get {self.network_class}")
        return network

    def create_optimization(self) -> BaseOptimization:
        """
        :return: BaseOptimization object from optimization_class and its parameters.
        """
        try:
            optimization = self.optimization_class(config=self.optimization_config)
        except:
            raise ValueError(
                f"[{self.__class__.__name__}] Given 'optimization_class' got an unexpected keyword argument 'config'")
        if not isinstance(optimization, BaseOptimization):
            raise TypeError(f"[{self.__class__.__name__}] Wrong 'optimization_class' type: BaseOptimization required, "
                            f"get {self.optimization_class}")
        return optimization

    def create_data_transformation(self) -> DataTransformation:
        """
        :return: DataTransformation object from data_transformation_class and its parameters.
        """
        try:
            data_transformation = self.data_transformation_class(network_config=self)
        except:
            raise ValueError(
                f"[{self.__class__.__name__}] Given 'data_transformation_class' got an unexpected keyword argument "
                f"'config'")
        if not isinstance(data_transformation, DataTransformation):
            raise TypeError(
                f"[{self.__class__.__name__}] Wrong 'data_transformation_class' type: DataTransformation required, "
                f"get {self.data_transformation_class}")
        return data_transformation

    def __str__(self) -> str:
        """
        :return: String containing information about the BaseDatasetConfig object
        """
        # Todo: fields in Configs are the set in Managers or objects, then remove __str__ method
        description = "\n"
        description += f"{self.__class__.__name__}\n"
        description += f"    Network class: {self.network_class.__name__}\n"
        description += f"    Optimization class: {self.optimization_class.__name__}\n"
        description += f"    Training materials: {self.training_stuff}\n"
        description += f"    Network directory: {self.network_dir}\n"
        description += f"    Which network: {self.which_network}\n"
        description += f"    Save each epoch: {self.save_each_epoch}\n"
        return description
