from DeepPhysX_Core.Manager.DatasetManager import DatasetManager
from DeepPhysX_Core.Manager.EnvironmentManager import EnvironmentManager

from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX_Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig

class DataManager:

    def __init__(self, dataset_config: BaseDatasetConfig, environment_config: BaseEnvironmentConfig,
                 session_name='default', session_dir=None, new_session=True,
                 training=True, record_data=None):

        self.is_training = training
        self.dataset_manager = None
        self.network_manager = None
        self.allow_dataset_fetch = True
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
            create_dataset = record_data is not None and (record_data[0] or record_data[1])

        # Create dataset if required
        if create_dataset:
            self.dataset_manager = DatasetManager(dataset_config=dataset_config, session_name=session_name,
                                                  session_dir=session_dir, new_session=new_session,
                                                  train=self.is_training, record_data=record_data)
        # Create environment if required
        if create_environment is None:  # If None then the dataset_manager exists
            create_environment = self.dataset_manager.requireEnvironment()
        if create_environment:
            self.environment_manager = EnvironmentManager(environment_config=environment_config)

    def getData(self, epoch=0, batch_size=1, animate=True):
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
                data = self.dataset_manager.getData(batch_size=batch_size, get_inputs=True, get_outputs=True, force_dataset_reload=True)
        # Prediction
        else:
            # Get data from environment
            data = self.environment_manager.getData(batch_size=batch_size, animate=animate, get_inputs=True, get_outputs=True)
            # Record data
            if self.dataset_manager is not None:
                self.dataset_manager.addData(data)
        return data

    def close(self):
        if self.environment_manager is not None:
            self.environment_manager.close()
        if self.dataset_manager is not None:
            self.dataset_manager.close()
