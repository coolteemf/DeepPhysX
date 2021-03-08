import os

from .DatasetManager import DatasetManager
from .NetworkManager import NetworkManager
from .StatsManager import StatsManager
import DeepPhysX.utils.pathUtils as pathUtils


class Manager:

    def __init__(self, session_name, network_config, trainer,
                 manager_dir=None, dataset_dir=None, network_dir=None,
                 partition_size=1, shuffle_dataset=True, save_each_epoch=False):

        self.sessionName = session_name
        self.networkConfig = network_config
        self.partitionSize = partition_size
        self.shuffleDataset = shuffle_dataset
        self.saveEachEpoch = save_each_epoch
        self.loadOnly = False

        # Trainer: must create a new session to avoid overwriting
        if trainer:
            # Create manager directory from the session name
            self.managerDir = os.path.join(pathUtils.getFirstCaller(), self.sessionName)
            # Avoid unwanted overwritten data
            self.managerDir = pathUtils.createDir(self.managerDir, key=self.sessionName)
            self.generateData = True if dataset_dir is None else False
            self.trainNetwork = True

        # Runner: load an existing session or create a custom new one
        else:
            # Create a custom session
            if manager_dir is None:
                # Must at least give a network path
                if network_dir is None:
                    raise Warning("You must at least give a network directory to the Runner. Shutting down.")
                self.trainNetwork = False
                # Create manager directory from the session name
                self.managerDir = os.path.join(pathUtils.getFirstCaller(), self.sessionName)
                self.managerDir = pathUtils.createDir(self.managerDir, key=self.sessionName)
                self.generateData = True if dataset_dir is None else False
            # Work in a full existing session
            else:
                self.managerDir = manager_dir
                dataset_dir = os.path.join(self.managerDir, 'dataset/')
                network_dir = os.path.join(self.managerDir, 'network/')
                self.generateData = False
                self.trainNetwork = False
                self.loadOnly = True

        # Create managers
        self.datasetManager = self.createDatasetManager() if dataset_dir is None else self.createDatasetManager(
            dataset_dir)
        self.networkManager = self.createNetworkManager() if network_dir is None else self.createNetworkManager(
            network_dir)

    def createDatasetManager(self, dataset_dir=None):
        existing_dataset = False if dataset_dir is None else True
        if dataset_dir is None:
            dataset_dir = os.path.join(self.managerDir, 'dataset/')
        return DatasetManager(session_name=self.sessionName, dataset_dir=dataset_dir,
                              existing_dataset=existing_dataset, manager_dir=self.managerDir,
                              partition_size=self.partitionSize, shuffle_dataset=self.shuffleDataset,
                              generate_data=self.generateData, load_only=self.loadOnly)

    def createNetworkManager(self, network_dir=None):
        existing_network = False if network_dir is None else True
        if network_dir is None:
            network_dir = os.path.join(self.managerDir, 'network/')
        return NetworkManager(session_name=self.sessionName, network_dir=network_dir,
                              existing_network=existing_network, manager_dir=self.managerDir,
                              train_network=self.trainNetwork, save_each_epoch=self.saveEachEpoch,
                              load_only=self.loadOnly)

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
