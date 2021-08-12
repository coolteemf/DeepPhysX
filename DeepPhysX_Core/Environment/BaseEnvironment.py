import numpy as np

from DeepPhysX_Core.AsyncSocket.TcpIpClient import TcpIpClient


class BaseEnvironment(TcpIpClient):

    def __init__(self, instance_id=1):
        """
        BaseEnvironment is an environment class to compute simulated data for the network and its optimization process.

        :param int instance_id: ID of the instance
        """

        super().__init__(instance_id=instance_id)
        print("Init base environment")

        # Environment data
        self.input, self.output = np.array([]), np.array([])
        self.input_size, self.output_size = None, None

    def create(self):
        """
        Create the environment given the configuration. Must be implemented by user.

        :return:
        """
        raise NotImplementedError

    def step(self):
        """
        Compute the number of steps specified by simulations_per_step

        :return:
        """
        raise NotImplementedError

    def computeInput(self):
        """
        Compute the data to be given as an input to the network.

        :return:
        """
        raise NotImplementedError

    def computeOutput(self):
        """
        Compute the data to be given as a ground truth to the network.

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

    def getInput(self):
        """
        :return: Input data
        """
        return self.input

    def getOutput(self):
        """
        :return: Ground truth data
        """
        return self.output

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

    def __str__(self):
        """
        :return: String containing information about the BaseEnvironmentConfig object
        """
        description = "\n"
        description += f"{self.name}\n"
        description += f"    Name: {self.name}\n"
        description += f"    Input size: {self.input_size}\n"
        description += f"    Output size: {self.output_size}\n"
        return description
