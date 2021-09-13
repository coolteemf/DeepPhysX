import os
import numpy as np

from DeepPhysX_Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig


class EnvironmentManager:

    def __init__(self, environment_config: BaseEnvironmentConfig, data_manager=None, visualizer_manager=None,
                 session_dir=None, batch_size=1):
        """
        Deals with the online generation of data for both training and running of the neural networks

        :param BaseEnvironmentConfig environment_config:
        :param DataManager data_manager: DataManager that handles the EnvironmentManager
        :param str session_dir: Name of the directory in which to write all of the necessary data
        """

        self.name = self.__class__.__name__

        self.data_manager = data_manager
        self.session_dir = session_dir
        self.number_of_thread = environment_config.number_of_thread
        # Create single or multiple environments according to multiprocessing value
        self.server = environment_config.createServer(environment_manager=self, batch_size=batch_size)

        # Init visualizer
        self.visualizer_manager = visualizer_manager
        self.initVisualizer()

        self.always_create_data = environment_config.always_create_data
        self.use_prediction_in_environment = environment_config.use_prediction_in_environment
        self.simulations_per_step = environment_config.simulations_per_step
        self.max_wrong_samples_per_step = environment_config.max_wrong_samples_per_step

    def getDataManager(self):
        """
        :return: DataManager that handle The EnvironmentManager
        """
        return self.data_manager

    def initVisualizer(self):
        if self.visualizer_manager is not None:
            data_dict = self.server.data_dict
            self.visualizer_manager.initView(data_dict)

    def step(self):
        """
        Trigger a step in Environments.
        :return:
        """
        self.getData(get_inputs=False, get_outputs=False, animate=True)

    def getData(self, get_inputs=True, get_outputs=True, animate=True):
        """
        Compute a batch of data from Environments.

        :param bool get_inputs: If True, compute and return input
        :param bool get_outputs: If True, compute and return output
        :param bool animate: If True, triggers an environment step
        :return: dictionnary containing all labeled data sent by the clients in their own dictionnary + in and out key corresponding to the batch
        """
        batch, data_dict = self.server.getBatch(get_inputs, get_outputs, animate)
        # if self.visualizer_manager is not None:
        #     self.visualizer_manager.updateFromBatch(data_dict)

        training_data = {'in': np.array(batch[0]) if get_inputs else np.array([]),
                         'out': np.array(batch[1]) if get_outputs else np.array([])}
        if 'loss' in data_dict.keys():
            training_data['loss'] = data_dict['loss']

        return training_data

    def updateVisualizer(self, visualization_data, id):
        self.visualizer_manager.updateFromSample(visualization_data, id)

    def applyPrediction(self, prediction):
        self.server.applyPrediction(prediction)

    def dispatchBatch(self, batch, get_inputs=True, get_outputs=True, animate=True):
        self.server.setDatasetBatch(batch)
        return self.getData(get_inputs=get_inputs, get_outputs=get_outputs, animate=animate)

    def close(self):
        """
        Close the environment

        :return:
        """
        self.server.close()

    def __str__(self):
        """
        :return: A string containing valuable information about the EnvironmentManager
        """
        description = "\n"
        description += f"# {self.name}\n"
        description += f"    Always create data: {self.always_create_data}\n"
        # description += f"    Record wrong samples: {self.record_wrong_samples}\n"
        description += f"    Number of threads: {self.number_of_thread}\n"
        # description += f"    Managed objects: Environment: {self.environment.env_name}\n"
        # Todo: manage the print log of each Environment since they can have different parameters
        # description += str(self.environment)
        return description
