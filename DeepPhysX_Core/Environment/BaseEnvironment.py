import numpy as np


class BaseEnvironment:

    def __init__(self, config, idx_instance=1):
        """
        BaseEnvironment is an environment class to compute simulated data for the network and its optimization process.

        :param config: BaseEnvironmentConfig.BaseEnvironmentProperties class containing BaseEnvironment parameters
        :param idx_instance:
        """

        self.name = self.__class__.__name__ + f"n°{idx_instance}"

        # Step variables
        self.simulations_per_step = config.simulations_per_step
        self.max_wrong_samples_per_step = config.max_wrong_samples_per_step

        # Visualizer is created just after the environment in BaseEnvironmentConfig
        self.visualizer = None

        # Environment data
        self.input, self.output = np.array([]), np.array([])
        self.input_size, self.output_size = None, None

        self.config = config

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

        :param bool check_input:
        :param bool check_output:
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

    def initVisualizer(self):
        """
        Set the actors to be rendered with the visualizer during simulation.

        :return:
        """
        pass

    def renderVisualizer(self):
        """
        Trigger the rendering method of the Visualizer.

        :return:
        """
        if self.visualizer is not None:
            self.visualizer.render()

    def save_wrong_sample(self, session_dir):
        """
        Save the current sample as an array. Can be reloaded with SampleVisualizer.

        :param str session_dir: Working directory
        :return:
        """
        if self.visualizer is not None:
            self.visualizer.saveSample(session_dir)

    def __str__(self):
        """
        :return: String containing information about the BaseEnvironmentConfig object
        """
        description = "\n"
        description += f"{self.name}\n"
        description += f"    Name: {self.name}\n"
        description += f"    Simulations per step: {self.simulations_per_step}\n"
        description += f"    Max wrong samples per step: {self.max_wrong_samples_per_step}\n"
        description += f"    Input size: {self.input_size}\n"
        description += f"    Output size: {self.output_size}\n"
        return description
