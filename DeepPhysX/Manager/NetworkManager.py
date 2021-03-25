import os
import shutil

import DeepPhysX.utils.pathUtils as pathUtils

from DeepPhysX.Network.NetworkConfig import NetworkConfig


class NetworkManager:

    def __init__(self, session_name, network_config: NetworkConfig, manager_dir, trainer):

        self.networkConfig = network_config

        self.networkDir = os.path.join(manager_dir, 'network/') if network_config.networkDir is None else network_config.networkDir
        self.managerDir = manager_dir
        self.networkTemplateName = session_name + '_network_{}.pth'

        self.existingNetwork = network_config.existingNetwork
        self.training = self.checkTraining(trainer, network_config.trainingMaterials)

        self.saveEachEpoch = network_config.saveEachEpoch
        self.savedCounter = 0

        self.network = None
        self.optimization = None
        self.setNetwork()

        self.description = ""

    def checkTraining(self, trainer, training_materials):
        if trainer and not training_materials:
            print("NetworkManager: You called Trainer without loss and optimizer. Shutting down.")
            quit(0)
        elif trainer:
            return True
        return False

    def setNetwork(self):
        self.network = self.networkConfig.createNetwork()
        self.optimization = self.networkConfig.createOptimization()
        if self.optimization.loss is not None:
            self.optimization.setLoss()
        # If training mode
        if self.training:
            self.optimization.setOptimizer(self.network)
            self.network.setTrain()
            # Re-train an existing network, copy directory
            if self.existingNetwork:
                shutil.copytree(self.networkDir, os.path.join(self.managerDir, 'network/'))
                self.networkDir = os.path.join(self.managerDir, 'network/')
            # Create a new network directory
            else:
                self.networkDir = pathUtils.createDir(self.networkDir, key='network')
        # If predict only
        else:
            self.network.setEval()
            # Need an existing network
            if not self.existingNetwork:
                print("NetworkManager: Need an existing network for prediction only. Shutting down")
                quit(0)
            # Reference the existing network
            else:
                os.symlink(self.networkDir, os.path.join(self.managerDir, 'network/'))
                networks_list = [os.path.join(self.networkDir, f) for f in os.listdir(self.networkDir) if
                                 os.path.isfile(os.path.join(self.networkDir, f)) and f.endswith('.pth')]
                which_network = self.networkConfig.whichNetwork
                if len(networks_list) > 1 and which_network == -1:
                    print("There is more than one network in this directory, loading the first one by default.")
                    print("If you want to load another one please use the 'which_network' variable.")
                    which_network = 0
                if which_network > len(networks_list) > 0:
                    print("The selected network doesn't exist (index is too big), loading the first one by default.")
                    which_network = 0
                if len(networks_list) == 0:
                    print("There is no network in {}. Shutting down.".format(self.networkDir))
                    quit(0)
                self.network.loadParameters(networks_list[which_network])

    def optimizeNetwork(self, prediction, ground_truth):
        loss = self.optimization.computeLoss(prediction, ground_truth)
        self.optimization.optimize(loss)
        return loss

    def computeLoss(self, prediction, ground_truth):
        return self.optimization.computeLoss(prediction, ground_truth)

    def saveNetwork(self, last_save=False, suffix=None):
        if last_save:
            path = self.networkDir + "network.pth"
        elif suffix is not None:
            path = self.networkDir + self.networkTemplateName.format(suffix)
        else:
            path = self.networkDir + self.networkTemplateName.format(self.savedCounter)
            self.savedCounter += 1
        print("Saving network at {}.".format(path))
        self.network.saveParameters(path)

    def close(self):
        if self.training:
            self.saveNetwork(last_save=True)
        del self.network
        del self.networkConfig

    def getDescription(self, minimal=False):
        if len(self.description) == 0:
            self.description += "\nNETWORK MANAGER:\n"
            nb_param = self.network.nbParameters()
            # Todo : move network description to network
            self.description += "   Number of parameters : {}\n".format(nb_param)
            # Cast nb_param to weight in bit(float = 32) to weight in Go(1bit = 1.25e-10Go)
            self.description += "   Weight in Go : {}\n".format(nb_param * 32 * 1.25e-10)
            self.description += "   Configuration : {}\n".format(self.networkConfig)
            self.description += "   Network : {}\n".format(self.network.type if minimal else self.network)
            self.description += "   Optimizer : {}, Learning rate : {}\n".format(self.optimization.optimizer,
                                                                                 self.optimization.lr)
            # self.description += "   Loss function : {}\n".format(str(self.loss).split(" ")[1])
            self.description += "   Loss function : {}\n".format(self.optimization.loss)
            self.description += "   Save each epoch : {}\n".format(self.saveEachEpoch)
        return self.description




