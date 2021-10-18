from numpy import array, empty, concatenate
from DeepPhysX_Core.Manager.VisualizerManager import VisualizerManager


class EnvironmentManager:

    def __init__(self,
                 environment_config=None,
                 data_manager=None,
                 session_dir=None,
                 batch_size=1,
                 train=True):

        """
        Deals with the online generation of data for both training and running of the neural networks

        :param BaseEnvironmentConfig environment_config:
        :param DataManager data_manager: DataManager that handles the EnvironmentManager
        :param str session_dir: Name of the directory in which to write all of the necessary data
        :param int batch_size: Number of samples in a batch of data
        :param bool train: True if this session is a network training
        """

        self.name = self.__class__.__name__

        self.data_manager = data_manager
        self.session_dir = session_dir
        self.number_of_thread = environment_config.number_of_thread
        # Create single or multiple environments according to multiprocessing value
        self.server = environment_config.createServer(environment_manager=self, batch_size=batch_size) if environment_config.as_tcpip_client else None
        self.environment = environment_config.createEnvironment(environment_manager=self) if not environment_config.as_tcpip_client else None
        self.batch_size = batch_size
        self.train = train

        # Init visualizer
        if environment_config.visual_object is None:
            self.visualizer_manager = None
        else:
            self.visualizer_manager = VisualizerManager(data_manager=data_manager, visual_object=environment_config.visual_object)
            self.initVisualizer()

        self.always_create_data = environment_config.always_create_data
        self.use_prediction_in_environment = environment_config.use_prediction_in_environment
        self.simulations_per_step = environment_config.simulations_per_step
        self.max_wrong_samples_per_step = environment_config.max_wrong_samples_per_step

        self.prediction_requested = False

    def getDataManager(self):
        """
        Return the Manager of the EnvironmentManager.

        :return: DataManager that handle The EnvironmentManager
        """
        return self.data_manager

    def initVisualizer(self):
        if self.visualizer_manager is not None:
            if self.server is not None:
                data_dict = self.server.visu_dict
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
        :return: Dictionary containing all labeled data sent by the clients in their own dictionary + in and out key
        corresponding to the batch
        """
        if self.server is not None:
            return self.getDataFromServer(get_inputs, get_outputs, animate)
        if self.environment is not None:
            return self.getDataFromEnvironment(get_inputs, get_outputs, animate)
        raise ValueError("[EnvironmentManager] There is no way to produce data.")

    def getDataFromServer(self, get_inputs, get_outputs, animate):
        """
        Compute a batch of data from Environments requested through TcpIpServer.

        :param bool get_inputs: If True, compute and return input
        :param bool get_outputs: If True, compute and return output
        :param bool animate: If True, triggers an environment step
        :return: Dictionary containing all labeled data sent by the clients in their own dictionary + in and out key
        corresponding to the batch
        """
        batch, data_dict = self.server.getBatch(get_inputs, get_outputs, animate)
        # if self.visualizer_manager is not None:
        #     self.visualizer_manager.updateFromBatch(data_dict)

        training_data = {'input': array(batch[0]) if get_inputs else array([]),
                         'output': array(batch[1]) if get_outputs else array([])}

        if 'loss' in data_dict:
            training_data['loss'] = data_dict['loss']

        return training_data

    def getDataFromEnvironment(self, get_inputs, get_outputs, animate):
        """
        Compute a batch of data directly from Environment.

        :param bool get_inputs: If True, compute and return input
        :param bool get_outputs: If True, compute and return output
        :param bool animate: If True, triggers an environment step
        :return: Dictionary containing all labeled data sent by the clients in their own dictionary + in and out key
        corresponding to the batch
        """
        inputs = empty((0, *self.environment.input_size)) if get_inputs else array([])
        input_condition = lambda input_array: input_array.shape[0] < self.batch_size if get_inputs else lambda _: False
        outputs = empty((0, *self.environment.output_size)) if get_outputs else array([])
        output_condition = lambda output_array: output_array.shape[0] < self.batch_size if get_outputs else lambda _: False
        data_dict = {}

        while input_condition(inputs) and output_condition(outputs):
            self.prediction_requested = False
            if animate:
                for current_step in range(self.simulations_per_step):
                    if current_step != self.simulations_per_step - 1:
                        self.environment.compute_essential_data = False
                        self.environment.step()
                    else:
                        self.environment.compute_essential_data = True
                        self.environment.step()
            if self.environment.checkSample() or not self.train:
                if get_inputs:
                    inputs = concatenate((inputs, self.environment.input[None, :]))
                if get_outputs:
                    outputs = concatenate((outputs, self.environment.output[None, :]))
                # received_data_dict = self.environment.data_dict
                # for key in received_data_dict:
                #     data_dict[key] = concatenate((data_dict[key], received_data_dict[key][None, :])) \
                #         if key in data_dict else array([received_data_dict[key]])
            else:
                # Record wrong sample
                # if self.data_manager is not None and self.data_manager.visualizer_manager is not None:
                #     self.data_manager.visualizer_manager.saveSample(self.session_dir)
                pass
        training_data = {'input': inputs,
                         'output': outputs}
        if 'loss' in data_dict.keys():
            training_data['loss'] = data_dict['loss']
        return training_data

    def requestPrediction(self, network_input):
        """
        Get a prediction of the network.

        :param network_input: Input of the network
        :return: Prediction of the network
        """
        self.prediction_requested = True
        return self.data_manager.manager.network_manager.computeOnlinePrediction(network_input)

    def updateVisualizer(self, visualization_data, index):
        """
        Update visualization.

        :param visualization_data: Dictionary containing visualization fields
        :param index: Index of the client
        :return:
        """
        self.visualizer_manager.updateFromSample(visualization_data, index)

    def applyPrediction(self, prediction):
        """
        Apply the prediction in the environment.

        :param prediction: Network prediction
        :return:
        """
        if self.server and not self.prediction_requested:
            self.server.applyPrediction(prediction)
        if self.environment and not self.prediction_requested:
            self.environment.applyPrediction(prediction)

    def dispatchBatch(self, batch):
        """
        Send samples from dataset to the Environments. Get back the training data.

        :param batch: Batch of samples.
        :return: Batch of training data.
        """
        self.server.setDatasetBatch(batch)
        return self.getData(get_inputs=False, get_outputs=False, animate=True)

    def close(self):
        """
        Close the environment

        :return:
        """
        if self.server:
            self.server.close()
        if self.environment:
            self.environment.close()

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
