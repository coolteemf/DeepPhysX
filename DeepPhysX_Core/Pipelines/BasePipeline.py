from typing import Dict, Optional, Any, List, Union

from DeepPhysX_Core.Network.BaseNetworkConfig import BaseNetworkConfig
from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX_Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX_Core.Manager.Manager import Manager
from DeepPhysX_Core.Manager.NetworkManager import NetworkManager
from DeepPhysX_Core.Manager.DataManager import DataManager
from DeepPhysX_Core.Manager.StatsManager import StatsManager
from DeepPhysX_Core.Manager.DatasetManager import DatasetManager
from DeepPhysX_Core.Manager.EnvironmentManager import EnvironmentManager
from DeepPhysX_Core.Manager.VisualizerManager import VisualizerManager


class BasePipeline:
    """
    | Base class defining Pipelines common variables.

    :param BaseNetworkConfig network_config: Specialisation containing the parameters of the network manager
    :param BaseDatasetConfig dataset_config: Specialisation containing the parameters of the dataset manager
    :param BaseEnvironmentConfig environment_config: Specialisation containing the parameters of the environment manager
    :param str session_name: Name of the newly created directory if session_dir is not defined
    :param Optional[str] session_dir: Name of the directory in which to write all the necessary data
    :param Optional[str] pipeline: Values at either 'training' or 'prediction'
    """

    def __init__(self,
                 network_config: Union[tuple, BaseNetworkConfig],
                 environment_config: Union[tuple, BaseEnvironmentConfig],
                 dataset_config: Union[tuple, BaseDatasetConfig],
                 session_name: str = 'default',
                 session_dir: Optional[str] = None,
                 pipeline: Optional[str] = None):

        self.name: str = self.__class__.__name__

        # Check the arguments
        # Network variables
        if isinstance(network_config, tuple):
            self.network_config = network_config[0](**network_config[1])
        else:
            self.network_config = network_config
        if network_config is not None and not isinstance(network_config, BaseNetworkConfig):
            raise TypeError(f"[{self.name}] The network configuration must be a BaseNetworkConfig")
        # Simulation variables
        if isinstance(environment_config, tuple):
            self.environment_config = environment_config[0](**environment_config[1])
        else:
            self.environment_config = environment_config
        if environment_config is not None and not isinstance(environment_config, BaseEnvironmentConfig):
            raise TypeError(f"[{self.name}] The environment configuration must be a BaseEnvironmentConfig")
        # Dataset variables
        if isinstance(dataset_config, tuple):
            self.dataset_config = dataset_config[0](**dataset_config[1])
        else:
            self.dataset_config = dataset_config
        if dataset_config is not None and not isinstance(dataset_config, BaseDatasetConfig):
            raise TypeError(f"[{self.name}] The dataset configuration must be a BaseDatasetConfig")
        if type(session_name) != str:
            raise TypeError(f"[{self.name}] The network config must be a BaseNetworkConfig object.")
        if session_dir is not None and type(session_dir) != str:
            raise TypeError(f"[{self.name}] The session directory must be a str.")

        self.type: str = pipeline    # Either training or prediction
        self.debug: bool = False
        self.new_session: bool = True
        self.record_data: Optional[Dict[str, bool]] = None  # Can be of type {'in': bool, 'out': bool}

        # Main manager
        self.manager: Optional[Manager] = None

    def get_any_manager(self, manager_names: Union[str, List[str]]) -> Optional[Any]:
        """
        | Return the desired Manager associated with the pipeline if it exists.

        self.manager = None

        :param Union[str, List[str]] manager_names: Name of the desired Manager or order of access to the desired
                                                    Manager
        :return: Manager associated with the Pipeline
        """

        # If manager variable is not defined, cannot access other manager
        if self.manager is None:
            return None

        # Direct access to manager
        if type(manager_names) == str:
            return getattr(self.manager, manager_names) if hasattr(self.manager, manager_names) else None

        # Intermediates to access manager
        accessed_manager = self.manager
        for next_manager in manager_names:
            if hasattr(accessed_manager, next_manager):
                accessed_manager = getattr(accessed_manager, next_manager)
            else:
                return None
        return accessed_manager

    def get_network_manager(self) -> NetworkManager:
        """
        | Return the NetworkManager associated with the pipeline.

        :return: NetworkManager associated with the pipeline
        """

        return self.get_any_manager('network_manager')

    def get_data_manager(self) -> DataManager:
        """
        | Return the DataManager associated with the pipeline.

        :return: DataManager associated with the pipeline
        """

        return self.get_any_manager('data_manager')

    def get_stats_manager(self) -> StatsManager:
        """
        | Return the StatsManager associated with the pipeline.

        :return: StatsManager associated with the pipeline
        """

        return self.get_any_manager('stats_manager')

    def get_dataset_manager(self) -> DatasetManager:
        """
        | Return the DatasetManager associated with the pipeline.

        :return: DatasetManager associated with the pipeline
        """

        return self.get_any_manager(['data_manager', 'dataset_manager'])

    def get_environment_manager(self) -> EnvironmentManager:
        """
        | Return the EnvironmentManager associated with the pipeline.

        :return: EnvironmentManager associated with the pipeline
        """

        return self.get_any_manager(['data_manager', 'environment_manager'])

    def get_visualizer_manager(self) -> VisualizerManager:
        """
        | Return the VisualizerManager associated with the pipeline.

        :return: VisualizerManager associated with the pipeline
        """

        return self.get_any_manager(['data_manager', 'environment_manager', 'visualizer_manager'])



