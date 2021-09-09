import os
import numpy as np

from DeepPhysX_Core.Network.BaseNetworkConfig import BaseNetworkConfig
import DeepPhysX_Core.utils.pathUtils as pathUtils


class NetworkManager:

    def __init__(self, network_config: BaseNetworkConfig, manager=None, session_name='default', session_dir=None, new_session=True,
                 train=True):
        """
        Deals with all the interactions with the neural network. Predictions, saves, initialisation, loading,
        back-propagation, etc...

        :param BaseNetworkConfig network_config: Specialisation containing the parameters of the network manager
        :param BaseDatasetConfig dataset_config: Specialisation containing the parameters of the dataset manager
        :param BaseEnvironmentConfig environment_config: Specialisation containing the parameters of the environment manager
        :param Manager manager : Manager that handle the network manager
        :param str session_name: Name of the newly created directory if session_dir is not defined
        :param str session_dir: Name of the directory in which to write all of the neccesary data
        :param bool new_session: Define the creation of new directories to store data
        :param bool train: If True prediction will cause tensors gradient creation
        """
        # Checking arguments
        if not isinstance(network_config, BaseNetworkConfig):
            raise TypeError("[NETWORKMANAGER] The network config must be a BaseNetworkConfig object.")
        if type(session_name) != str:
            raise TypeError("[NETWORKMANAGER] The session name must be a str.")
        if session_dir is not None and type(session_dir) != str:
            raise TypeError("[NETWORKMANAGER] The session directory must be a str.")
        if type(train) != bool:
            raise TypeError("[NETWORKMANAGER] The 'train' argument' must be a boolean.")

        self.manager = manager

        self.network_config = network_config

        self.session_dir = session_dir if session_dir is not None else os.path.join(pathUtils.getFirstCaller(),
                                                                                    session_name)
        network_dir = network_config.network_dir
        self.network_dir = network_dir if network_dir is not None else os.path.join(self.session_dir, 'network/')
        self.network_template_name = session_name + '_network_{}'

        self.existing_network = network_config.existing_network or os.path.exists(self.network_dir)
        if train and not network_config.training_stuff:
            raise ValueError("[NETWORKMANAGER] You are training without loss and optimizer. Shutting down.")
        self.training = train

        self.save_each_epoch = network_config.save_each_epoch
        self.saved_counter = 0

        self.network = None
        self.optimization = None
        self.setNetwork()
        self.data_in = None
        self.data_out = None
        self.data_gt = None

        self.description = ""

    def getManager(self):
        """

        :return: Manager that handles the NetworkManager
        """
        return self.manager

    def setNetwork(self):
        """
        Set the network to the corresponding weight from a given file

        :return:
        """
        self.network = self.network_config.createNetwork()
        self.network.setDevice()
        self.data_transformation = self.network_config.createDataTransformation()
        self.optimization = self.network_config.createOptimization()
        self.optimization.manager = self
        if self.optimization.loss_class is not None:
            self.optimization.setLoss()
        # If training mode
        if self.training:
            self.network.setTrain()
            self.optimization.setOptimizer(self.network)
            # Re-train an existing network, copy directory
            if self.existing_network:
                self.network_dir = pathUtils.copyDir(self.network_dir, self.session_dir, dest_dir='network')
            # Create a new network directory
            else:
                self.network_dir = pathUtils.createDir(self.network_dir, check_existing='network')
        # If predict only
        else:
            print("eval")
            self.network.setEval()
            # Need an existing network
            if not self.existing_network:
                print("NetworkManager: Need an existing network for prediction only. Shutting down")
                quit(0)
            # Reference the existing network
            else:
                # Get eventual epoch saved networks
                networks_list = [os.path.join(self.network_dir, f) for f in os.listdir(self.network_dir) if
                                 os.path.isfile(os.path.join(self.network_dir, f)) and f.__contains__('_network_.')]
                networks_list = sorted(networks_list)
                # Add the final saved network
                last_saved_network = [os.path.join(self.network_dir, f) for f in os.listdir(self.network_dir) if
                                      os.path.isfile(os.path.join(self.network_dir, f)) and f.__contains__('network.')]
                networks_list = networks_list + last_saved_network
                which_network = self.network_config.which_network
                if len(networks_list) == 0:
                    print("NetworkManager: There is no network in {}. Shutting down.".format(self.network_dir))
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

    def computePredictionAndLoss(self, batch, optimize):
        """
        Make a prediction with the data passe as argument, optimize or not the network

        :param dict batch:  Format {'in': numpy.ndarray, 'out': numpy.ndarray} Contains the input value and ground truth
        to compare against
        :param bool optimize: If true run a back propagation

        :return:
        """
        # Getting data from the data manager
        data_in, data_gt = self.network.transformFromNumpy(batch['in']), self.network.transformFromNumpy(batch['out'])
        loss_data = self.network.transformFromNumpy(batch['loss']) if 'loss' in batch.keys() else None

        # Compute prediction
        data_in = self.data_transformation.transformBeforePrediction(data_in)
        data_out = self.network.predict(data_in)

        # Compute loss
        data_out, data_gt = self.data_transformation.transformBeforeLoss(data_out, data_gt)
        loss_dict = self.optimization.computeLoss(data_out.reshape(data_gt.shape), data_gt, loss_data)
        # Optimizing network if training
        if optimize:
            self.optimization.optimize()
        # Transform prediction to be compatible with environment
        data_out = self.data_transformation.transformBeforeApply(data_out)
        prediction = self.network.transformToNumpy(data_out)
        return prediction, loss_dict

    def computeOnlinePrediction(self, network_input, compute_gradient=False):
        """
        Make a prediction with the data passe as argument, optimize or not the network

        :param bool compute_gradient: Compute gradient when predicting
        :param numpy.ndarray network_input: Input of the network=

        :return:
        """
        # Getting data from the data manager
        data_in = self.network.transformFromNumpy(np.copy(network_input))

        # Compute prediction
        data_in = self.data_transformation.transformBeforePrediction(data_in)
        data_in.requires_grad = compute_gradient
        pred = self.network.predict(data_in)
        pred, _ = self.data_transformation.transformBeforeLoss(pred, pred)
        pred = self.data_transformation.transformBeforeApply(pred)
        pred = self.network.transformToNumpy(pred)
        pred = np.array(pred, dtype=float)
        return pred.reshape(-1)

    def saveNetwork(self, last_save=False):
        """
        Save the network with the corresponding suffix so they do not erase the last save.

        :param bool last_save: Do not add suffix if it's the last save

        :return:
        """
        if last_save:
            path = self.network_dir + "network"
            print(f"Saving network at {path}.pth")
            self.network.saveParameters(path)
        elif self.save_each_epoch:
            path = self.network_dir + self.network_template_name.format(self.saved_counter)
            self.saved_counter += 1
            print(f"Saving network at {path}.pth")
            self.network.saveParameters(path)

    def close(self):
        """
        Closing procedure.
        :return:
        """
        if self.training:
            self.saveNetwork(last_save=True)
        del self.network
        del self.network_config

    def __str__(self):
        """
        :return: A string containing valuable information about the NetworkManager
        """
        if len(self.description) == 0:
            self.description += "\nNETWORK MANAGER:\n"
            nb_param = self.network.nbParameters()
            # Todo : move network description to network
            self.description += "   Number of parameters : {}\n".format(nb_param)
            # Cast nb_param to weight in bit(float = 32) to weight in Go(1bit = 1.25e-10Go)
            self.description += "   Weight in Go : {}\n".format(nb_param * 32 * 1.25e-10)
            self.description += "   Configuration : {}\n".format(self.network_config)
            self.description += "   Network : {}\n".format(self.network)
            self.description += "   Optimizer : {}, Learning rate : {}\n".format(self.optimization.optimizer,
                                                                                 self.optimization.lr)
            # self.description += "   Loss function : {}\n".format(str(self.loss).split(" ")[1])
            self.description += "   Loss function : {}\n".format(self.optimization.loss_dict)
            self.description += "   Save each epoch : {}\n".format(self.save_each_epoch)
        return self.description
