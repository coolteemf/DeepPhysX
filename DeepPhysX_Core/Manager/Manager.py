import os

from DeepPhysX_Core.Pipelines.BasePipeline import BasePipeline

from DeepPhysX_Core.Manager.DataManager import DataManager
from DeepPhysX_Core.Manager.NetworkManager import NetworkManager
from DeepPhysX_Core.Manager.StatsManager import StatsManager

from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX_Core.Network.BaseNetworkConfig import BaseNetworkConfig
from DeepPhysX_Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig

import DeepPhysX_Core.utils.pathUtils as pathUtils


class Manager:

    def __init__(self, pipeline: BasePipeline, network_config: BaseNetworkConfig, dataset_config: BaseDatasetConfig,
                 environment_config: BaseEnvironmentConfig, session_name='default', session_dir=None, new_session=True):
        """
        Collection of all the specialized managers. Allows for some basic functions call. More specific behaviour have to
        be directly call from the corresponding manager

        :param BasePipeline pipeline: Specialisation That define the type of usage
        :param BaseNetworkConfig network_config: Specialisation containing the parameters of the network manager
        :param BaseDatasetConfig dataset_config: Specialisation containing the parameters of the dataset manager
        :param BaseEnvironmentConfig environment_config: Specialisation containing the parameters of the environment manager
        :param str session_name: Name of the newly created directory if session_dir is not defined
        :param str session_dir: Name of the directory in which to write all of the neccesary data
        :param bool new_session: Define the creation of new directories to store data
        """
        # Trainer: must create a new session to avoid overwriting
        if pipeline.type == 'training':
            train = True
            # Create manager directory from the session name
            self.session_dir = os.path.join(pathUtils.getFirstCaller(), session_name)
            # Avoid unwanted overwritten data
            if new_session:
                self.session_dir = pathUtils.createDir(self.session_dir, check_existing=session_name)
        # Prediction: work in an existing session
        elif pipeline.type == 'prediction':
            train = False
            # Find the session directory with the name
            if session_dir is None:
                if session_name is None:
                    raise ValueError("[Manager] Prediction needs at least the session directory or the session name.")
                self.session_dir = os.path.join(pathUtils.getFirstCaller(), session_name)
            # Find the session name with the directory
            else:
                self.session_dir = session_dir
            if not os.path.exists(self.session_dir):
                raise ValueError("[Manager] The session directory {} does not exists.".format(self.session_dir))

        else:
            raise ValueError("[Manager] The pipeline must be either training or prediction.")

        self.session_name = (session_name if session_name is not None else os.path.basename(session_dir)).split("/")[-1]
        # Always create the network manager (man it's DEEP physics here...)
        self.network_manager = NetworkManager(network_config=network_config, session_name=self.session_name,
                                              session_dir=self.session_dir, new_session=new_session, train=train)

        # Always create the data manager for same reason
        self.data_manager = DataManager(dataset_config=dataset_config, environment_config=environment_config,
                                        session_name=self.session_name, session_dir=self.session_dir, new_session=new_session,
                                        training=train, record_data=pipeline.record_data)

        # Create the stats manager for training
        self.stats_manager = StatsManager(log_dir=os.path.join(self.session_dir, 'stats/')) if train else None

    def getData(self, epoch=0, batch_size=1, animate=True):
        """
        Fetch data from the DataManager

        :param int epoch: Epoch ID
        :param int batch_size: Size of a batch
        :param bool animate: If True allows to run environment step

        :return:
        """
        self.data_manager.getData(epoch=epoch, batch_size=batch_size, animate=animate)

    def optimizeNetwork(self):
        """
        Compute a prediction and run a back propagation with the current batch

        :return: tuple (numpy.ndarray, float)
        """
        prediction, loss = self.network_manager.computePrediction(self.data_manager.data, optimize=True)
        return prediction, loss

    def getPrediction(self):
        """
        Compute a prediction with the current batch

        :return: tuple (numpy.ndarray, float)
        """
        prediction, loss = self.network_manager.computePrediction(self.data_manager.data, optimize=False)
        return prediction, loss

    def saveNetwork(self):
        """
        Save network weights as a pth file

        :return:
        """
        self.network_manager.saveNetwork()

    def close(self):
        """
        Call all managers close procedure

        :return:
        """
        if self.data_manager is not None:
            self.data_manager.close()
        if self.network_manager is not None:
            self.network_manager.close()
        if self.stats_manager is not None:
            self.stats_manager.close()

    def __str__(self):
        """
        :return: A string containing valuable information about the Managers
        """
        manager_description = ""
        if self.data_manager is not None:
            manager_description += str(self.data_manager)
        if self.network_manager is not None:
            manager_description += str(self.network_manager)
        if self.stats_manager is not None:
            manager_description += str(self.stats_manager)
        return manager_description
