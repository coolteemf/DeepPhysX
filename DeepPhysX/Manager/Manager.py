import os

from .DatasetManager import DatasetManager
from .NetworkManager import NetworkManager
from .EnvironmentManager import EnvironmentManager
# Todo: add statsManager
from .StatsManager import StatsManager
import DeepPhysX.utils.pathUtils as pathUtils


class Manager:

    def __init__(self, session_name, network_config, dataset_config, trainer, environment_config=None,
                 manager_dir=None):

        self.sessionName = session_name
        self.networkConfig = network_config
        self.datasetConfig = dataset_config

        # Trainer: must create a new session to avoid overwriting
        if trainer:
            # Create manager directory from the session name
            self.managerDir = os.path.join(pathUtils.getFirstCaller(), self.sessionName)
            # Avoid unwanted overwritten data
            self.managerDir = pathUtils.createDir(self.managerDir, key=self.sessionName)
        # Runner: load an existing session or create a custom new one
        else:
            # Create a custom session
            if manager_dir is None:
                # Must at least give a network path
                if self.networkConfig.networkDir is None:
                    raise Warning("You must at least give a network directory to the Runner. Shutting down.")
                # Create manager directory from the session name
                self.managerDir = os.path.join(pathUtils.getFirstCaller(), self.sessionName)
                self.managerDir = pathUtils.createDir(self.managerDir, key=self.sessionName)
                # Work in a full existing session
            # Todo: disputable
            else:
                self.managerDir = manager_dir
                self.datasetConfig.datasetDir = os.path.join(self.managerDir, 'dataset/')
                self.datasetConfig.existingDataset = True
                self.networkConfig.networkDir = os.path.join(self.managerDir, 'network/')
                self.networkConfig.existingNetwork = True
        # Create managers
        self.datasetManager = DatasetManager(session_name=self.sessionName, dataset_config=self.datasetConfig,
                                             manager_dir=self.managerDir, trainer=trainer)
        self.networkManager = NetworkManager(session_name=self.sessionName, network_config=self.networkConfig,
                                             manager_dir=self.managerDir, trainer=trainer)
        self.environmentManager = EnvironmentManager(environment_config=environment_config)
        # Todo: manage conflict if environment and datasetDir are set

    def getData(self, source, get_inputs, get_outputs, batch_size):
        if source == 'environment':
            if self.environmentManager is None:
                print("Manager: Trying to get data from an non existing environment. Shutting down.")
                quit(0)
            data = self.environmentManager.getData(get_inputs, get_outputs, batch_size)
            self.datasetManager.addData(data)
            return data
        if source == 'dataset':
            data = self.datasetManager.getData(get_inputs, get_outputs, batch_size)
            return data

    def close(self):
        if self.datasetManager is not None:
            self.datasetManager.close()
        if self.networkManager is not None:
            self.networkManager.close()

    def getDescription(self):
        manager_description = ""
        if self.networkManager is not None:
            # Todo: add minimal description
            manager_description = self.networkManager.description()
        if self.datasetManager is not None:
            # Todo: add minimal description
            manager_description += self.datasetManager.description()
        return manager_description
