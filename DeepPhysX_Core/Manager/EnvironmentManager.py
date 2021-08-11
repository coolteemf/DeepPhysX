import os
import numpy as np
import multiprocessing as mp

from DeepPhysX_Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig


class EnvironmentManager:

    # Todo: remove batch_size from inputs
    def __init__(self, environment_config: BaseEnvironmentConfig, data_manager=None, session_dir=None, batch_size=1):
        """
        Deals with the online generation of data for both training and running of the neural networks

        :param BaseEnvironmentConfig environment_config:
        :param DataManager datamanager: DataManager that handles the EnvironmentManager
        :param str session_dir: Name of the directory in which to write all of the neccesary data
        """
        self.data_manager = data_manager
        self.session_dir = session_dir
        self.number_of_thread = environment_config.number_of_thread
        # Create single or multiple environments according to multiprocessing value
        # Todo: if prediction, force nb_thread = 1 !!!
        if environment_config.number_of_thread == 1:
            self.environment = environment_config.createEnvironment(environment_manager=self)
        else:
            # batch_size = self.data_manager.manager.pipeline.batch_size
            self.server = environment_config.createServer(environment_manager=self, batch_size=batch_size)

        self.always_create_data = environment_config.always_create_data
        self.simulations_per_step = environment_config.simulations_per_step
        self.max_wrong_samples_per_step = environment_config.max_wrong_samples_per_step

    def getDataManager(self):
        """

        :return: DataManager that handle The EnvironmentManager
        """
        return self.data_manager

    def getData(self, batch_size, get_inputs, get_outputs, animate):
        # Getting data from single environment: batch is built from samples
        if self.number_of_thread == 1:
            data = self.buildBatchFromEnvironment(batch_size, get_inputs, get_outputs, animate)
        # Getting data from server: batch is built in lower level
        else:
            data = self.getBatchFromServer(get_inputs, get_outputs)
        return data

    def buildBatchFromEnvironment(self, batch_size, get_inputs, get_outputs, animate):
        """
        Compute a batch of data from the environment

        :param int batch_size: Size of a batch
        :param bool get_inputs: If True compute and return input
        :param bool get_outputs: If True compute an return output
        :param bool animate: If True run environment step

        :return: dict of format {'in': numpy.ndarray, 'out': numpy.ndarray}
        """

        inputs = np.empty((0, *self.environment.input_size)) if get_inputs else np.array([])
        input_condition = lambda x: x.shape[0] <= batch_size if get_inputs else lambda x: True

        outputs = np.empty((0, *self.environment.output_size)) if get_outputs else np.array([])
        output_condition = lambda x: x.shape[0] <= batch_size if get_outputs else lambda x: True

        while input_condition(inputs) and output_condition(outputs):
            if animate:
                for _ in range(self.simulations_per_step):
                    self.environment.step()

            if get_inputs:
                self.environment.computeInput()
                if self.environment.checkSample(check_input=get_inputs, check_output=False):
                    inputs = np.concatenate((inputs, self.environment.getInput()[None, :]))
                else:
                    if self.getDataManager().visualizer_manager is not None:
                        self.getDataManager().visualizer_manager.saveSample(self.session_dir)

            if get_outputs:
                self.environment.computeOutput()
                if self.environment.checkSample(check_input=False, check_output=get_outputs):
                    outputs = np.concatenate((outputs, self.environment.getOutput()[None, :]))
                else:
                    if self.getDataManager().visualizer_manager is not None:
                        self.getDataManager().visualizer_manager.saveSample(self.session_dir)
        return {'in': inputs, 'out': outputs}

    def getBatchFromServer(self, get_inputs, get_outputs):
        batch = self.server.getBatch(get_inputs, get_outputs)
        inputs = np.array(batch[0]) if get_inputs else np.array([])
        outputs = np.array(batch[1]) if get_outputs else np.array([])
        return {'in': inputs, 'out': outputs}

    def close(self):
        """
        Close the environment

        :return:
        """
        if self.number_of_thread == 1:
            self.environment.close()
        else:
            self.server.close()

    def __str__(self):
        """
        :return: A string containing valuable information about the EnvironmentManager
        """
        environment_manager_str = "Environment Manager :\n"
        return environment_manager_str + "    Environment description \n" + f"{str(self.environment)}"
