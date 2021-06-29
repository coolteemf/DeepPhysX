import os

from DeepPhysX.Pipelines.BasePipeline import BasePipeline
from DeepPhysX.Manager.DatasetManager import DatasetManager
from DeepPhysX.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX.Manager.NetworkManager import NetworkManager
from DeepPhysX.Network.BaseNetworkConfig import BaseNetworkConfig
from DeepPhysX.Manager.EnvironmentManager import EnvironmentManager
from DeepPhysX.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX.Manager.StatsManager import StatsManager
import DeepPhysX.utils.pathUtils as pathUtils


class Manager:

    def __init__(self, pipeline: BasePipeline, network_config: BaseNetworkConfig, dataset_config: BaseDatasetConfig,
                 environment_config: BaseEnvironmentConfig, session_name='default', session_dir=None, new_session=True,
                 stats_window=50):

        # Todo: checking the arguments
        self.record_data = pipeline.record_data

        # Trainer: must create a new session to avoid overwriting
        if pipeline.type == 'training':
            self.train = True
            create_environment = None
            create_dataset = True
            # Create manager directory from the session name
            self.session_dir = os.path.join(pathUtils.getFirstCaller(), session_name)
            # Avoid unwanted overwritten data
            if new_session:
                self.session_dir = pathUtils.createDir(self.session_dir, check_existing=session_name)

        # Prediction: work in an existing session
        elif pipeline.type == 'prediction':
            self.train = False
            create_environment = True
            create_dataset = pipeline.record_data['in']
            # Find the session directory with the name
            if session_dir is None:
                if session_name is None:
                    raise ValueError("[Manager] Prediction needs at least the session directory or the session name.")
                self.session_dir = os.path.join(pathUtils.getFirstCaller(), session_name)
            # Find the session name with the directory
            else:
                self.session_dir = session_dir
                session_name = session_name if session_name is not None else os.path.basename(session_dir)
            if not os.path.exists(self.session_dir):
                raise ValueError("[Manager] The session directory {} does not exists.".format(self.session_dir))

        else:
            raise ValueError("[Manager] The pipeline must be either training or prediction.")

        # Create the dataset manager for training or for prediction when recording data
        self.dataset_manager = DatasetManager(dataset_config=dataset_config, session_name=session_name,
                                              session_dir=self.session_dir, new_session=new_session, train=self.train,
                                              record_data=self.record_data) if create_dataset else None

        # Create the environment manager for prediction or for training when dataset does not exists or partially exists
        if create_environment is None:
            create_environment = self.dataset_manager.create_environment
        if create_environment:
            self.environment_manager = EnvironmentManager(environment_config=environment_config)
            self.always_create_data = environment_config.always_create_data
        else:
            self.environment_manager = None

        # Always create the network manager (man it's DEEP physics here...)
        self.network_manager = NetworkManager(network_config=network_config, session_name=session_name,
                                              session_dir=self.session_dir, new_session=new_session, train=self.train)

        # Create the stats manager for training
        self.stats_manager = StatsManager(log_dir=os.path.join(self.session_dir, 'stats/'),
                                          sliding_window_size=stats_window) if self.train else None

    def getData(self, epoch=0, batch_size=1, animate=True):
        # Training
        if self.train:
            # Get data from environment if used and if the data should be created at this epoch
            if (self.environment_manager is not None) and (epoch == 0 or self.always_create_data):
                data = self.environment_manager.getData(batch_size=batch_size, animate=animate,
                                                        get_inputs=True, get_outputs=True)
                self.dataset_manager.addData(data)
            # Get data from the dataset
            else:
                data = self.dataset_manager.getData(batch_size=batch_size, get_inputs=True, get_outputs=True)
        # Prediction
        else:
            # Get data from environment
            data = self.environment_manager.getData(batch_size=batch_size, animate=animate, get_inputs=True,
                                                    get_outputs=True)
            # Record data
            if self.dataset_manager is not None:
                self.dataset_manager.addData(data)
        # Send data to the network
        self.network_manager.setData(data)

    def optimizeNetwork(self):
        prediction, ground_truth = self.network_manager.computePrediction()
        return self.network_manager.optimizeNetwork(prediction, ground_truth)

    def getPrediction(self):
        prediction, ground_truth = self.network_manager.computePrediction()
        loss = self.network_manager.computeLoss(prediction, ground_truth)
        return self.network_manager.network.transformToNumpy(prediction), loss

    def saveNetwork(self):
        self.network_manager.saveNetwork()

    def close(self):
        if self.dataset_manager is not None:
            self.dataset_manager.close()
        if self.network_manager is not None:
            self.network_manager.close()
        if self.environment_manager is not None:
            self.environment_manager.close()
        # if self.stats_manager is not None:
        #     self.stats_manager.close()

    def getDescription(self):
        manager_description = ""
        if self.network_manager is not None:
            # Todo: add minimal description
            manager_description = self.network_manager.description()
        if self.dataset_manager is not None:
            # Todo: add minimal description
            manager_description += self.dataset_manager.description()
        return manager_description
