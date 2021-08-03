from DeepPhysX_Core.Network.BaseNetworkConfig import BaseNetworkConfig
from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX_Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig



class BasePipeline:

    def __init__(self, network_config: BaseNetworkConfig, dataset_config: BaseDatasetConfig,
                 environment_config: BaseEnvironmentConfig, session_name='default', session_dir=None,
                 pipeline=None):
        """
        Base class defining Pipelines common variables

        :param BaseNetworkConfig network_config: Specialisation containing the parameters of the network manager
        :param BaseDatasetConfig dataset_config: Specialisation containing the parameters of the dataset manager
        :param BaseEnvironmentConfig environment_config: Specialisation containing the parameters of the environment manager
        :param visualizer_class: Visualization class from which an instance will be created
        :type visualizer_class: type[BaseVisualizer]
        :param str session_name: Name of the newly created directory if session_dir is not defined
        :param str session_dir: Name of the directory in which to write all of the neccesary data
        :param bool new_session: Define the creation of new directories to store data
        :param str pipeline: Values at either 'training' or 'prediction'
        """

        # Check the arguments
        if not isinstance(network_config, BaseNetworkConfig):
            raise TypeError("[BaseRunner] The network configuration must be a BaseNetworkConfig")
        if not isinstance(environment_config, BaseEnvironmentConfig):
            raise TypeError("[BaseRunner] The environment configuration must be a BaseEnvironmentConfig")
        if dataset_config is not None and not isinstance(dataset_config, BaseDatasetConfig):
            raise TypeError("[BaseRunner] The dataset configuration must be a BaseDatasetConfig")
        if type(session_name) != str:
            raise TypeError("[BaseRunner] The network config must be a BaseNetworkConfig object.")
        if session_dir is not None and type(session_dir) != str:
            raise TypeError("[BaseRunner] The session directory must be a str.")

        self.type = pipeline    # Either training or prediction
        self.new_session = True
        self.record_data = None  # Can be of type {'in': bool, 'out': bool}
        # Dataset variables
        self.dataset_config = dataset_config

        # Network variables
        self.network_config = network_config

        # Simulation variables
        self.environment_config = environment_config

        self.manager = None

    def getNetworkManager(self):
        """
        :return: The NetworkManager associated with the pipeline
        """
        return self.manager.network_manager

    def getDataManager(self):
        """
        :return: The DataManager associated with the pipeline
        """
        return self.manager.data_manager

    def getStatsManager(self):
        """
        :return: The StatsManager associated with the pipeline
        """
        return self.manager.stats_manager

    def getDatasetManager(self):
        """
        :return: The DatasetManager associated with the pipeline
        """
        return self.manager.data_manager.dataset_manager

    def getEnvironmentManager(self):
        """
        :return: The EnvironmentManager associated with the pipeline
        """
        return self.manager.data_manager.environment_manager

    def getVisualizerManager(self):
        """
        :return: The VisualizerManager associated with the pipeline
        """
        return self.manager.data_manager.visualizer_manager



