import os

from .DatasetManager import DatasetManager
from .NetworkManager import NetworkManager
from .EnvironmentManager import EnvironmentManager
from .StatsManager import StatsManager
import DeepPhysX.utils.pathUtils as pathUtils


class Manager:

    def __init__(self, session_name, network_config, dataset_config, trainer, environment_config=None,
                 manager_dir=None, stats_window=50):

        self.sessionName = session_name
        self.networkConfig = network_config
        self.datasetConfig = dataset_config
        self.environmentConfig = environment_config

        # Trainer: must create a new session to avoid overwriting
        if trainer:
            # Create manager directory from the session name
            self.managerDir = os.path.join(pathUtils.getFirstCaller(), self.sessionName)
            # Avoid unwanted overwritten data
            self.managerDir = pathUtils.createDir(self.managerDir, key=self.sessionName)
            # Set other dir if they are None
            if self.datasetConfig.datasetDir is None:
                self.datasetConfig.datasetDir = os.path.join(self.managerDir, 'dataset/')
            if self.networkConfig.networkDir is None:
                self.networkConfig.networkDir = os.path.join(self.managerDir, 'network/')
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
        if environment_config is None:
            self.environmentManager = None
        else:
            self.environmentManager = EnvironmentManager(environment_config=environment_config)
        # Todo: manage conflict if environment and datasetDir are set
        self.statsManager = StatsManager(log_dir=os.path.join(self.managerDir, 'stats/'),
                                         sliding_window_size=stats_window)

    def getData(self, epoch, batch_size=1, get_inputs=True, get_outputs=True):
        if (self.environmentManager is not None) and (epoch == 0 or self.environmentConfig.alwaysCreateData):
            data = self.environmentManager.getData(batch_size=batch_size, get_inputs=get_inputs, get_outputs=get_outputs)
            self.datasetManager.addData(data)
        else:
            data = self.datasetManager.getData(batch_size=batch_size, get_inputs=get_inputs, get_outputs=get_outputs)
        self.networkManager.setData(data)

    def optimizeNetwork(self):
        prediction, ground_truth = self.networkManager.computePrediction()
        return self.networkManager.optimizeNetwork(prediction, ground_truth)

    def saveNetwork(self):
        self.networkManager.saveNetwork()

    def close(self):
        if self.datasetManager is not None:
            self.datasetManager.close()
        if self.networkManager is not None:
            self.networkManager.close()
        if self.environmentManager is not None:
            self.environmentManager.close()
        if self.statsManager is not None:
            self.statsManager.close()

    def getDescription(self):
        manager_description = ""
        if self.networkManager is not None:
            # Todo: add minimal description
            manager_description = self.networkManager.description()
        if self.datasetManager is not None:
            # Todo: add minimal description
            manager_description += self.datasetManager.description()
        return manager_description




    def getPrediction(self):
        return self.networkManager.getPrediction()

    def computeLoss(self, prediction, ground_truth):
        return self.networkManager.computeLoss(prediction, ground_truth)
