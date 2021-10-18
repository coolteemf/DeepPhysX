from numpy import array

from DeepPhysX_Core.AsyncSocket.TcpIpClient import TcpIpClient


class BaseEnvironment(TcpIpClient):

    def __init__(self,
                 instance_id=1,
                 number_of_instances=1,
                 as_tcpip_client=True,
                 ip_address='localhost',
                 port=10000,
                 visual_object=None,
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
        self.visual_object = visual_object(visualizer=None) if visual_object is not None else None
        self.environment_manager = environment_manager
        self.sample_in, self.sample_out = None, None

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

    def checkSample(self, check_input=True, check_output=True):
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
        return {}

    def applyPrediction(self, prediction):
        """
        Apply network prediction in environment.

        :param prediction: Prediction data
        :return:
        """
        pass

    def setDatasetSample(self, sample_in, sample_out):
        """
        Set the sample received from Dataset.

        :param sample_in: Input sample
        :param sample_out: Associated output sample
        :return:
        """
        self.sample_in = sample_in
        self.sample_out = sample_out

    def setTrainingData(self, input_array, output_array):
        """
        Set the training data to send to the TcpIpServer.

        :param input_array: Network input
        :param output_array: Network expected output
        :return:
        """
        if self.compute_essential_data:
            if self.as_tcpip_client:
                self.sync_send_training_data(network_input=input_array, network_output=output_array)
            else:
                self.input = input_array
                self.output = output_array

    def getPrediction(self, input_array):
        """
        Request a prediction from Environment.

        :param input_array: Network input
        :return:
        """
        if self.as_tcpip_client:
            return self.sync_send_prediction_request(network_input=input_array)
        if self.environment_manager is not None:
            return self.environment_manager.requestPrediction(network_input=input_array)
        raise ValueError(f"[{self.name}] Can't get prediction since Environment has no Manager.")

    def setVisualizationData(self, visu_dict):
        """
        Set the updated visualization data.

        :param visu_dict: Dictionary containing visualization data fields
        :return:
        """
        if self.as_tcpip_client:
            self.sync_send_visualization_data(visu_dict)
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
