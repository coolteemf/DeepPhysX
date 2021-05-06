import os

from DeepPhysX.Network.BaseNetworkConfig import BaseNetworkConfig
import DeepPhysX.utils.pathUtils as pathUtils


class NetworkManager:

    def __init__(self, session_name, network_config: BaseNetworkConfig, manager_dir, trainer):

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
        self.network.setDevice()
        self.optimization = self.networkConfig.createOptimization()
        if self.optimization.loss_class is not None:
            self.optimization.setLoss()
        # If training mode
        if self.training:
            self.optimization.setOptimizer(self.network)
            self.network.setTrain()
            # Re-train an existing network, copy directory
            if self.existingNetwork:
                self.networkDir = pathUtils.copyDir(self.networkDir, self.managerDir, key='network')
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
                self.networkDir = pathUtils.copyDir(self.networkDir, self.managerDir, key='network')
                # Get eventual epoch saved networks
                networks_list = [os.path.join(self.networkDir, f) for f in os.listdir(self.networkDir) if
                                 os.path.isfile(os.path.join(self.networkDir, f)) and f.endswith('.pth') and
                                 f.__contains__('_network_')]
                networks_list = sorted(networks_list)
                # Add the final saved network
                last_saved_network = [os.path.join(self.networkDir, f) for f in os.listdir(self.networkDir) if
                                      os.path.isfile(os.path.join(self.networkDir, f)) and f.endswith('network.pth')]
                networks_list = networks_list + last_saved_network
                which_network = self.networkConfig.whichNetwork
                if len(networks_list) == 0:
                    print("NetworkManager: There is no network in {}. Shutting down.".format(self.networkDir))
                    quit(0)
                elif len(networks_list) == 1:
                    which_network = 0
                elif len(networks_list) > 1 and which_network is None:
                    print("There is more than one network in this directory, loading the most trained by default.")
                    print("If you want to load another network please use the 'which_network' variable.")
                    which_network = -1
                elif which_network > len(networks_list) > 1:
                    print("The selected network doesn't exist (index is too big), loading the most trained by default.")
                    which_network = -1
                print("NetworkManager: Loading network from {}.".format(networks_list[which_network]))
                self.network.loadParameters(networks_list[which_network])

    def setData(self, data):
        self.network.convertData(data)

    def computePrediction(self):
        self.network.transformInput()
        self.network.predict()
        self.network.transformPrediction()
        self.network.transformGroundTruth()
        return self.network.getPrediction(), self.network.getGroundTruth()

    def optimizeNetwork(self, prediction, ground_truth):
        print("NETWORK MANAGER DEBUG: prediction and ground truth data")
        print(type(prediction))
        print(type(ground_truth))
        loss = self.optimization.computeLoss(prediction, ground_truth)
        print("NETWORK MANAGER DEBUG: point reached")
        self.optimization.optimize()
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




