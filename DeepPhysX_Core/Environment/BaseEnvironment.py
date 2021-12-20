from numpy import array

from DeepPhysX_Core.AsyncSocket.TcpIpClient import TcpIpClient


class BaseEnvironment(TcpIpClient):

    def __init__(self,
                 ip_address='localhost',
                 port=10000,
                 instance_id=1,
                 number_of_instances=1,
                 as_tcpip_client=True,
                 environment_manager=None):
        """
        BaseEnvironment is an environment class to compute simulated data for the network and its optimization process.

        :param str ip_address: IP address of the TcpIpObject
        :param int port: Port number of the TcpIpObject
        :param int instance_id: ID of the instance
        :param int number_of_instances: Number of simultaneously launched instances
        :param bool as_tcpip_client: Environment is owned by a TcpIpClient if True, by an EnvironmentManager if False
        :param environment_manager: EnvironmentManager that handles the Environment if 'as_tcpip_client' is False
        """

        TcpIpClient.__init__(self,
                             instance_id=instance_id,
                             number_of_instances=number_of_instances,
                             as_tcpip_client=as_tcpip_client,
                             ip_address=ip_address,
                             port=port)

        # Input and output to give to the network
        self.input, self.output = array([]), array([])
        # Variables to store samples from Dataset
        self.sample_in, self.sample_out = None, None
        # Loss data
        self.loss_data = None
        # Manager if the Environment is not a TcpIpClient
        self.environment_manager = environment_manager

    def create(self):
        """
        Create the Environment.
        Must be implemented by user.

        :return:
        """
        raise NotImplementedError

    def init(self):
        """
        Initialize the Environment.
        Not mandatory.

        :return:
        """
        pass

    def step(self):
        """
        Compute the number of steps in the Environment specified by simulations_per_step in EnvironmentConfig.
        Must be implemented by user.

        :return:
        """
        raise NotImplementedError

    def check_sample(self, check_input=True, check_output=True):
        """
        Check if the current produced sample is usable for training.
        Not mandatory.

        :param bool check_input: If True, input tensors need to be checked
        :param bool check_output: If True, output tensors need to be checked
        :return: bool - Current data can be used or not
        """
        return True

    def close(self):
        """
        Close the Environment.
        Not mandatory.

        :return:
        """
        pass

    def recv_parameters(self, param_dict):
        """
        Exploit received parameters before scene creation.
        Not mandatory.

        :param dict param_dict: Dictionary of parameters
        :return:
        """
        pass

    def send_parameters(self):
        """
        Create a dictionary of parameters to send to the manager.
        Not mandatory.

        :return:
        """
        return

    def send_visualization(self):
        """
        Define the visualization objects to send to the Visualizer.
        Not mandatory.

        :return:
        """
        return

    def apply_prediction(self, prediction):
        """
        Apply network prediction in environment.
        Not mandatory.

        :param ndarray prediction: Prediction data
        :return:
        """
        pass

    def setDatasetSample(self, sample_in, sample_out, additional_in={}, additional_out={}):
        """
        Set the sample received from Dataset.

        :param ndarray sample_in: Input sample
        :param ndarray sample_out: Output sample
        :param dict additional_in: Contains each additional input data samples
        :param dict additional_out: Contains each additional output data samples
        :return:
        """

        # Network in / out samples
        self.sample_in = sample_in
        self.sample_out = sample_out
        # Additional in / out samples
        self.additional_inputs = additional_in if additional_in == {} else additional_in['additional_data']
        self.additional_outputs = additional_out if additional_out == {} else additional_out['additional_data']

    def setTrainingData(self, input_array, output_array):
        """
        Set the training data to send to the TcpIpServer or the EnvironmentManager.

        :param ndarray input_array: Network input
        :param ndarray output_array: Network expected output
        :return:
        """

        # Training data is set if the Environment can compute data
        if self.compute_essential_data:
            # Through TcpIp sending
            if self.as_tcpip_client:
                self.sync_send_training_data(network_input=input_array, network_output=output_array)
            # Directly get by EnvironmentManager
            else:
                self.input = input_array
                self.output = output_array

    def setLossData(self, loss_data):
        """
        Set the loss data to send to the TcpIpServer or the EnvironmentManager.

        :param loss_data: Optional data to compute loss.
        :return:
        """

        # Training data is set if the Environment can compute data
        if self.compute_essential_data:
            # Through TcpIp sending
            if self.as_tcpip_client:
                self.sync_send_labeled_data(loss_data, 'loss')
            # Directly get by EnvironmentManager
            else:
                self.loss_data = loss_data

    def additionalInDataset(self, label, data_array):
        """
        Set additional input data fields to store in the dataset.

        :param str label: Name of the data field
        :param data_array: Data to store
        :return:
        """

        self.additional_inputs[label] = data_array

    def additionalOutDataset(self, label, data_array):
        """
        Set additional output data fields to store in the dataset.

        :param str label: Name of the data field
        :param data_array: Data to store
        :return:
        """

        self.additional_outputs[label] = data_array

    def getPrediction(self, input_array):
        """
        Request a prediction from Network.

        :param ndarray input_array: Network input
        :return:
        """

        # If Environment is a TcpIpClient, send request to the Server
        if self.as_tcpip_client:
            return self.sync_send_prediction_request(network_input=input_array)

        # Otherwise, check the hierarchy of managers
        if self.environment_manager.data_manager is None:
            raise ValueError("Cannot request prediction if DataManager does not exist")
        elif self.environment_manager.data_manager.manager is None:
            raise ValueError("Cannot request prediction if Manager does not exist")
        elif self.environment_manager.data_manager.manager.network_manager is None:
            raise ValueError("Cannot request prediction if NetworkManager does not exist")
        # Get a prediction
        return self.environment_manager.data_manager.manager.network_manager.computeOnlinePrediction(
            network_input=input_array[None, ])

    async def get_prediction(self, input_array):
        """
        Request a prediction from Network.

        :param ndarray input_array: Network input
        :return:
        """

        # If Environment is a TcpIpClient, send request to the Server
        if self.as_tcpip_client:
            return await self.send_prediction_request(network_input=input_array)

        # Otherwise, check the hierarchy of managers
        if self.environment_manager.data_manager is None:
            raise ValueError("Cannot request prediction if DataManager does not exist")
        elif self.environment_manager.data_manager.manager is None:
            raise ValueError("Cannot request prediction if Manager does not exist")
        elif self.environment_manager.data_manager.manager.network_manager is None:
            raise ValueError("Cannot request prediction if NetworkManager does not exist")
        # Get a prediction
        return self.environment_manager.data_manager.manager.network_manager.computeOnlinePrediction(
            network_input=input_array[None, ])

    async def update_visualisation(self, visu_dict):
        """
        Triggers the Visualizer update.

        :param dict visu_dict: Updated visualization data.
        :return:
        """

        # If Environment is a TcpIpClient, request to the Server
        if self.as_tcpip_client:
            await self.send_visualization_data(visualization_data=visu_dict)
        # Otherwise, request to the EnvironmentManager
        else:
            self.environment_manager.visualizer.updateFromSample(visu_dict, self.instance_id)

    def sync_update_visualisation(self, visu_dict):
        """
        Triggers the Visualizer update.

        :param dict visu_dict: Updated visualization data.
        :return:
        """

        # If Environment is a TcpIpClient, request to the Server
        if self.as_tcpip_client:
            self.sync_send_visualization_data(visualization_data=visu_dict)
        # Otherwise, request to the EnvironmentManager
        else:
            self.environment_manager.visualizer.updateFromSample(visu_dict, self.instance_id)

    def __str__(self):
        """
        :return: String containing information about the BaseEnvironmentConfig object
        """
        description = "\n"
        description += f"  {self.name}\n"
        description += f"    Name: {self.name} n°{self.instance_id}\n"
        description += f"    Comments:\n"
        description += f"    Input size: {self.input_size}\n"
        description += f"    Output size: {self.output_size}\n"
        return description
