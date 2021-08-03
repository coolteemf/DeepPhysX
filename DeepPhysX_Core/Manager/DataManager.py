from DeepPhysX_Core.Manager.DatasetManager import DatasetManager
from DeepPhysX_Core.Manager.EnvironmentManager import EnvironmentManager

from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX_Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX_Core.Manager.VisualizerManager import VisualizerManager


class DataManager:

    def __init__(self, dataset_config: BaseDatasetConfig, environment_config: BaseEnvironmentConfig, manager=None,
                 visualizer_class=None, session_name='default', session_dir=None, new_session=True,
                 training=True, record_data=None):
        """
        DataManager deals with the generation of input / output tensors. His job is to call getData on either the
        DatasetManager or the EnvironmentManager according to the context.

        :param BaseDatasetConfig dataset_config: Specialisation containing the parameters of the dataset manager
        :param BaseEnvironmentConfig environment_config: Specialisation containing the parameters of the environment manager
        :param Manager manager: Manager that handle The DataManager
        :param visualizer_class: Visualization class from which an instance will be created
        :type visualizer_class: type[BaseVisualizer]
        :param str session_name: Name of the newly created directory if session_dir is not defined
        :param str session_dir: Name of the directory in which to write all of the neccesary data
        :param bool new_session: Define the creation of new directories to store data
        :param bool training: True if this session is a network training
        :param dict record_data: Format {\'in\': bool, \'out\': bool} save the tensor when bool is True
        """
        self.manager = manager
        self.is_training = training
        self.dataset_manager = None
        self.network_manager = None
        self.allow_dataset_fetch = True
        self.data = None
        self.visualizer_manager = None if visualizer_class is None else VisualizerManager(data_manager=self, visualizer_class=visualizer_class)
        # Training
        if self.is_training:
            # Always create a dataset_manager for training
            create_dataset = True
            # Create an environment if a) dataset in not existing b) dataset will be completed during the session
            create_environment = None
        # Prediction
        else:
            # Always create an environment for prediction
            create_environment = True
            # Create a dataset if data will be stored from environment during prediction
            create_dataset = record_data is not None and (record_data['in'] or record_data['out'])

        # Create dataset if required
        if create_dataset:
            self.dataset_manager = DatasetManager(data_manager=self, dataset_config=dataset_config, session_name=session_name,
                                                  session_dir=session_dir, new_session=new_session,
                                                  train=self.is_training, record_data=record_data)
        # Create environment if required
        if create_environment is None:  # If None then the dataset_manager exists
            create_environment = self.dataset_manager.requireEnvironment()
        if create_environment:
            self.environment_manager = EnvironmentManager(data_manager=self, environment_config=environment_config,
                                                          session_dir=session_dir)

    def getManager(self):
        """

        :return: Manager that handle The DataManager
        """
        return self.manager

    def getData(self, epoch=0, batch_size=1, animate=True):
        """
        Fetch data from EnvironmentManager or DatasetManager according to the context

        :param int epoch: Current epoch ID
        :param int batch_size: Size of the desired batch
        :param int animate: Allow EnvironmentManager to generate a new sample

        :return:
        """
        # Training
        if self.is_training:
            data = None
            # Try to fetch data from the dataset
            if self.allow_dataset_fetch:
                data = self.dataset_manager.getData(batch_size=batch_size, get_inputs=True, get_outputs=True)
            # If data could not be fetch, try to generate them from the environment
            # Get data from environment if used and if the data should be created at this epoch
            if data is None and self.environment_manager is not None and (epoch == 0 or self.environment_manager.always_create_data):
                self.allow_dataset_fetch = False
                data = self.environment_manager.getData(batch_size=batch_size, animate=animate, get_inputs=True, get_outputs=True)
                # We create a partition to write down the data in the case it's not already existing.
                if self.dataset_manager.current_in_partition is None:
                    self.dataset_manager.createNewPartitions()
                self.dataset_manager.addData(data)
            # Force data from the dataset
            else:
                data = self.dataset_manager.getData(batch_size=batch_size, get_inputs=True, get_outputs=True, force_partition_reload=True)
        # Prediction
        else:
            # Get data from environment
            data = self.environment_manager.getData(batch_size=batch_size, animate=animate, get_inputs=True, get_outputs=True)
            # Record data
            if self.dataset_manager is not None:
                self.dataset_manager.addData(data)
        self.data = data

    def close(self):
        """
        Launch the closing procedure on its managers

        :return:
        """
        if self.environment_manager is not None:
            self.environment_manager.close()
        if self.dataset_manager is not None:
            self.dataset_manager.close()

    def __str__(self):
        """
        :return: A string containing valuable information about the DataManager
        """
        data_manager_str = "DataManager handles : \n"
        if self.environment_manager:
            data_manager_str += str(self.environment_manager)
        if self.dataset_manager:
            data_manager_str += str(self.dataset_manager)
        return data_manager_str

