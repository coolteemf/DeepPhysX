from numpy import array

from DeepPhysX_Core.AsyncSocket.TcpIpClient import TcpIpClient, BytesNumpyConverter
from DeepPhysX_Core.Visualizer.MeshVisualizer import MeshVisualizer


class BaseEnvironment(TcpIpClient):

    def __init__(self, ip_address='localhost', port=10000, data_converter=BytesNumpyConverter, instance_id=1, number_of_instances=1,
                 visualizer_class=MeshVisualizer):
        """
        BaseEnvironment is an environment class to compute simulated data for the network and its optimization process.

        :param int instance_id: ID of the instance
        """

        super(BaseEnvironment, self).__init__(ip_address=ip_address, port=port, data_converter=data_converter,
                                              instance_id=instance_id, number_of_instances=number_of_instances)
        self.input, self.output = array([]), array([])
        self.visualizer = visualizer_class()
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
        self.sample_in = sample_in
        self.sample_out = sample_out

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
