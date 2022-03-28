from typing import Any, Optional, Dict, Union
from numpy import ndarray

from DeepPhysX_Core.Manager.DatasetManager import DatasetManager
from DeepPhysX_Core.Manager.EnvironmentManager import EnvironmentManager
from DeepPhysX_Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig


class DataManager:
    """
    | DataManager deals with the generation of input / output tensors. His job is to call get_data on either the
      DatasetManager or the EnvironmentManager according to the context.

    :param Optional[BaseDatasetConfig] dataset_config: Specialisation containing the parameters of the dataset manager
    :param Optional[BaseEnvironmentConfig] environment_config: Specialisation containing the parameters of the
                                                               environment manager
    :param Optional[Any] manager: Manager that handle The DataManager
    :param str session_name: Name of the newly created directory if session_dir is not defined
    :param Optional[str] session_dir: Name of the directory in which to write all the necessary data
    :param bool new_session: Define the creation of new directories to store data
    :param bool training: True if this session is a network training
    :param Dict[str, bool] record_data: Format {\'in\': bool, \'out\': bool} save the tensor when bool is True
    :param int batch_size: Number of samples in a batch
    """

    def __init__(self,
                 dataset_config: Optional[Union[BaseDatasetConfig], tuple[2]] = None,
                 environment_config: Optional[Union[BaseEnvironmentConfig, tuple[2]]] = None,
                 manager: Optional[Any] = None,
                 session_name: str = 'default',
                 session_dir: Optional[str] = None,
                 new_session: bool = True,
                 training: bool = True,
                 record_data: Dict[str, bool] = None,
                 batch_size: int = 1):

        self.name: str = self.__class__.__name__

        self.manager: Optional[Any] = manager
        self.is_training: bool = training
        self.dataset_manager: Optional[DatasetManager] = None
        self.environment_manager: Optional[EnvironmentManager] = None
        self.allow_dataset_fetch: bool = True
        self.data: Optional[Dict[str, ndarray]] = None

        #We need to instantiate environment_config because its attributes are used afterwards
        if isinstance(environment_config, tuple):
            env_config = environment_config[0](**environment_config[1])
        else:
            env_config = environment_config
        if isinstance(dataset_config, tuple):
            data_config = dataset_config[0](**dataset_config[1])
        else:
            data_config = dataset_config

        # Training
        if self.is_training:
            # Always create a dataset_manager for training
            create_dataset = True
            # Create an environment if prediction must be applied else ask DatasetManager
            create_environment = False
            if env_config is not None:
                create_environment = None if not env_config.use_dataset_in_environment else True
        # Prediction
        else:
            # Always create an environment for prediction
            create_environment = True
            # Create a dataset if data will be stored from environment during prediction
            create_dataset = record_data is not None and (record_data['input'] or record_data['output'])

        # Create dataset if required
        if create_dataset:
            self.dataset_manager = DatasetManager(data_manager=self, dataset_config=data_config,
                                                  session_name=session_name, session_dir=session_dir,
                                                  new_session=new_session, train=self.is_training,
                                                  record_data=record_data)
        # Create environment if required
        if create_environment is None:  # If None then the dataset_manager exists
            create_environment = self.dataset_manager.new_dataset()
        if create_environment:
            self.environment_manager = EnvironmentManager(data_manager=self, environment_config=env_config,
                                                          batch_size=batch_size, train=self.is_training)

    def get_manager(self) -> Any:
        """
        | Return the manager of DataManager.

        :return: Manager that handle the DataManager
        """

        return self.manager

    def get_data(self, epoch: int = 0, batch_size: int = 1, animate: bool = True) -> Dict[str, ndarray]:
        """
        | Fetch data from EnvironmentManager or DatasetManager according to the context

        :param int epoch: Current epoch ID
        :param int batch_size: Size of the desired batch
        :param bool animate: Allow EnvironmentManager to generate a new sample

        :return: Newly computed data
        """

        # Training
        if self.is_training:
            data = None
            # Get data from environment if used and if the data should be created at this epoch
            if data is None and self.environment_manager is not None and \
                    (epoch == 0 or self.environment_manager.always_create_data) and self.dataset_manager.new_dataset():
                self.allow_dataset_fetch = False
                data = self.environment_manager.get_data(animate=animate, get_inputs=True, get_outputs=True)
                self.dataset_manager.add_data(data)
            # Force data from the dataset
            else:
                data = self.dataset_manager.get_data(batch_size=batch_size, get_inputs=True, get_outputs=True)
                if self.environment_manager is not None and self.environment_manager.use_dataset_in_environment:
                    new_data = self.environment_manager.dispatch_batch(batch=data)
                    if len(new_data['input']) != 0:
                        data['input'] = new_data['input']
                    if len(new_data['output']) != 0:
                        data['output'] = new_data['output']
                    if 'loss' in new_data:
                        data['loss'] = new_data['loss']
                elif self.environment_manager is not None:
                    # EnvironmentManager is no longer used
                    self.environment_manager.close()
                    self.environment_manager = None

        # Prediction
        else:
            # Get data from environment
            data = self.environment_manager.get_data(animate=animate, get_inputs=True, get_outputs=True)
            # Record data
            if self.dataset_manager is not None:
                self.dataset_manager.add_data(data)

        self.data = data
        return data

    def close(self) -> None:
        """
        | Launch the closing procedure on its managers
        """

        if self.environment_manager is not None:
            self.environment_manager.close()
        if self.dataset_manager is not None:
            self.dataset_manager.close()

    def __str__(self) -> str:
        """
        :return: A string containing valuable information about the DataManager
        """

        data_manager_str = ""
        if self.environment_manager:
            data_manager_str += str(self.environment_manager)
        if self.dataset_manager:
            data_manager_str += str(self.dataset_manager)
        return data_manager_str
