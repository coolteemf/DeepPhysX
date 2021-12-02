from numpy import array

from DeepPhysX_Core.AsyncSocket.TcpIpClient import TcpIpClient


class BaseEnvironment(TcpIpClient):

    def __init__(self,
                 instance_id=1,
                 number_of_instances=1,
                 as_tcpip_client=True,
<<<<<<< HEAD
                 ip_address='localhost',
                 port=10000,
                 visual_object=None,
=======
                 visualizer_class=None,
>>>>>>> Init parameter update
                 environment_manager=None):
        """
        BaseEnvironment is an environment class to compute simulated data for the network and its optimization process.

        :param int instance_id: ID of the instance
        :param int number_of_instances: Number of simultaneously launched instances
        :param as_tcpip_client: Environment is own by a TcpIpClient if True, by an EnvironmentManager if False
        :param ip_address: IP address of the TcpIpObject
        :param port: Port number of the TcpIpObject
        :param visual_object: VedoObject class to template visual data
        :param environment_manager: EnvironmentManager that handles the Environment if as_tcpip_client is False
        """

        TcpIpClient.__init__(self,
                             instance_id=instance_id,
                             number_of_instances=number_of_instances,
                             as_tcpip_client=as_tcpip_client,
                             ip_address=ip_address,
                             port=port)

        self.input, self.output = array([]), array([])
        self.visual_object = visualizer_class(visualizer=None) if visualizer_class is not None else None
        self.environment_manager = environment_manager
        self.sample_in, self.sample_out = None, None
<<<<<<< HEAD
        self.loss_data = None
=======
>>>>>>> Environment update

    def create(self):
        """
        Create the environment given the configuration. Must be implemented by user.

        :return:
        """
        raise NotImplementedError

    def init(self):
        """
        Initialize environment.

        :return:
        """
        pass

    def step(self):
        """
        Compute the number of steps specified by simulations_per_step

        :return:
        """
        raise NotImplementedError

    def check_sample(self, check_input=True, check_output=True):
        """
        Check if the current sample is an outlier.

        :param bool check_input: True if input tensor need to be checked
        :param bool check_output: True if output tensor need to be checked
        :return: Current data can be used or not.
        """
        return True

    def close(self):
        """
        Close the environment.

        :return:
        """
        pass

    def recv_parameters(self, param_dict):
        """
        Exploit received parameters before scene creation.

        :param dict param_dict: Dictionary of parameters
        :return:
        """
        pass

    def send_parameters(self):
        """
        Create a dictionary of parameters to send to the manager.

        :return: Dictionary of parameters
        """
        return

    def send_visualization(self):
        return

<<<<<<< HEAD
    def send_visualization(self):
        return {}

    def applyPrediction(self, prediction):
=======
    def apply_prediction(self, prediction):
>>>>>>> Environment update
        """
        Apply network prediction in environment.

        :param prediction: Prediction data
        :return:
        """
        pass

<<<<<<< HEAD
    def setDatasetSample(self, sample_in, sample_out, additional_in=None, additional_out=None):
        """
        Set the sample received from Dataset.

        :param additional_out:
        :param additional_in:
        :param sample_in: Input sample
        :param sample_out: Associated output sample
        :return:
        """
=======
    def set_dataset_sample(self, sample_in, sample_out):
>>>>>>> Environment update
        self.sample_in = sample_in
        self.sample_out = sample_out
        self.additional_inputs = additional_in
        self.additional_outputs = additional_out

<<<<<<< HEAD
    def setTrainingData(self, input_array, output_array):
        """
        Set the training data to send to the TcpIpServer.

        :param input_array: Network input
        :param output_array: Network expected output
        :return:
        """
=======
    def set_training_data(self, input_array, output_array):
>>>>>>> Environment update
        if self.compute_essential_data:
            if self.as_tcpip_client:
                self.sync_send_training_data(network_input=input_array, network_output=output_array)
            else:
                self.input = input_array
                self.output = output_array
                # Todo: same without server

    def setLossData(self, loss_data):
        """
        Set the loss data to send to the TcpIpServer.

        :param loss_data: Optional data to compute loss.
        :return:
        """
        if self.compute_essential_data:
            if self.as_tcpip_client:
                self.sync_send_labeled_data(loss_data, 'loss')
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

<<<<<<< HEAD
    def getPrediction(self, input_array):
        """
        Request a prediction from Environment.

        :param input_array: Network input
        :return:
        """
=======
    def sync_get_prediction(self, input_array):
>>>>>>> Environment update
        if self.as_tcpip_client:
            return self.sync_send_prediction_request(network_input=input_array)
        if self.environment_manager is not None:
            return self.environment_manager.requestPrediction(network_input=input_array)
        raise ValueError(f"[{self.name}] Can't get prediction since Environment has no Manager.")

<<<<<<< HEAD
    def setVisualizationData(self, visu_dict):
        """
        Set the updated visualization data.

        :param visu_dict: Dictionary containing visualization data fields
        :return:
        """
=======
    async def get_prediction(self, input_array):
        if self.as_tcpip_client:
            return await self.send_prediction_request(network_input=input_array)
        if self.environment_manager.data_manager is None:
            raise ValueError("Cannot request prediction if DataManager does not exist")
        elif self.environment_manager.data_manager.manager is None:
            raise ValueError("Cannot request prediction if Manager does not exist")
        elif self.environment_manager.data_manager.manager.network_manager is None:
            raise ValueError("Cannot request prediction if NetworkManager does not exist")
        else:
            return self.environment_manager.data_manager.manager.network_manager.computeOnlinePrediction(network_input=input_array[None, ])

    async def update_visualisation(self, visu_dict):
        if self.as_tcpip_client:
            await self.send_visualization_data(visualization_data=visu_dict)
        else:
            self.environment_manager.visualizer.updateFromSample(visu_dict, self.instance_id)

    def sync_update_visualisation(self, visu_dict):
>>>>>>> Environment update
        if self.as_tcpip_client:
            self.sync_send_visualization_data(visualization_data=visu_dict)
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
