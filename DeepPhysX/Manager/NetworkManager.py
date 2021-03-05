import os
import inspect
import shutil


class NetworkManager:

    def __init__(self, network_name, retrain_network=False, save_each_epoch=False):
        frm = inspect.stack()[1]
        mod = inspect.getmodule(frm[0])
        caller_path = os.path.dirname(os.path.abspath(mod.__file__))
        self.networkDir = os.path.join(caller_path, network_name, 'network/')
        self.retrainNetwork = retrain_network
        self.saveEachEpoch = save_each_epoch
        self.savedCount = 0
        self.networkTemplateName = "network_{}.pth"
        self.network = None
        self.config = None
        self.optimizer = None
        self.lr = None
        self.loss = None
        self.device = None
        self.description = ""

    def setNetwork(self, config, loss, optimizer=None, lr=None, which_network=1):
        self.config = config
        self.network = config.create()
        self.network.setDevice()
        # Check if we are using the network for prediction only
        if optimizer is None or lr is None or self.retrainNetwork:
            networks_list = [os.path.join(self.networkDir, f) for f in os.listdir(self.networkDir) if
                             os.path.isfile(os.path.join(self.networkDir, f)) and f.endswith('.pth')]
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
        # Create the folder were we will save the network
        else:
            if os.path.isdir(self.networkDir):
                shutil.rmtree(self.networkDir, ignore_errors=True)
            os.makedirs(self.networkDir)
            self.lr = lr
            # self.optimizer = optimizer(self.network.getParameters(), lr=lr)
            self.optimizer = optimizer
            self.loss = loss

    def saveNetwork(self, last_save=False, suffix=None):
        if last_save:
            path = self.networkDir + "network.pth"
        elif suffix is not None:
            path = self.networkDir + self.networkTemplateName.format(suffix)
        else:
            path = self.networkDir + self.networkTemplateName.format(self.savedCount)
            self.savedCount += 1
        print("Saving network at {}.".format(path))
        self.network.saveParameters(path)

    def close(self):
        if self.optimizer is not None:
            self.saveNetwork(last_save=True)
        del self.network
        del self.config

    def setSaveEachEpoch(self, save_each_epoch):
        self.saveEachEpoch = save_each_epoch

    def getDescription(self, minimal=False):
        if len(self.description) == 0:
            self.description += "\nNETWORK MANAGER:\n"
            nb_param = self.network.nbParameters()
            # Todo : move network description to network
            self.description += "   Number of parameters : {}\n".format(nb_param)
            # Cast nb_param to weight in bit(float = 32) to weight in Go(1bit = 1.25e-10Go)
            self.description += "   Weight in Go : {}\n".format(nb_param*32*1.25e-10)
            self.description += "   Device : {}\n".format(self.device)
            self.description += "   Configuration : {}\n".format(self.config)
            self.description += "   Network : {}\n".format(self.network.type if minimal else self.network)
            self.description += "   Optimizer : {}, Learning rate : {}\n".format(self.optimizer, self.lr)
            # self.description += "   Loss function : {}\n".format(str(self.loss).split(" ")[1])
            self.description += "   Loss function : {}\n".format(self.loss)
            self.description += "   Save each epoch : {}\n".format(self.saveEachEpoch)
        return self.description
