from numpy import array, empty, concatenate
from asyncio import run
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
        if environment_config.visualizer is None:
            self.visualizer_manager = None
        else:
            self.visualizer_manager = VisualizerManager(data_manager=data_manager,
                                                        visualizer=environment_config.visualizer,
                                                        screenshot_rate=environment_config.screenshot_sample_rate)
            self.initVisualizer()

        self.always_create_data = environment_config.always_create_data
        self.use_prediction_in_environment = environment_config.use_prediction_in_environment
        self.simulations_per_step = environment_config.simulations_per_step
        self.max_wrong_samples_per_step = environment_config.max_wrong_samples_per_step

        self.prediction_requested = False
        self.dataset_batch = None

    def getDataManager(self):
        """
        Return the Manager of the EnvironmentManager.

        :return: DataManager that handle The EnvironmentManager
        """
        return self.data_manager

    def initVisualizer(self):
        if self.visualizer_manager is not None:
            if self.server is not None:
                data_dict = {}
                for client_id in self.server.data_dict:
                    data_dict[client_id] = self.server.data_dict[client_id]['visualisation']
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
<<<<<<< HEAD
        :return: Dictionary containing all labeled data sent by the clients in their own dictionary + in and out key
        corresponding to the batch
=======
        :return: dictionary containing all labeled data sent by the clients in their own dictionnary + in and out key corresponding to the batch
>>>>>>> updated manager to remove useless server functionallities, changed function name too
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
        for key in ['dataset_in', 'dataset_out']:
            for field in data_dict[key]:
                data_dict[key][field] = array(data_dict[key][field])
            training_data[key] = data_dict[key]

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
        inputs, outputs = [], []
        input_condition = lambda x: len(x) < self.batch_size if get_inputs else lambda _: False
        output_condition = lambda x: len(x) < self.batch_size if get_outputs else lambda _: False
        data_dict = {}
        # Produce batch
        while input_condition(inputs) and output_condition(outputs):
            self.prediction_requested = False

            if self.dataset_batch is not None:
                sample_in = self.dataset_batch['input'][0]
                self.dataset_batch['input'] = self.dataset_batch['input'][1:]
                sample_out = self.dataset_batch['output'][0]
                self.dataset_batch['output'] = self.dataset_batch['output'][1:]
                additional_in, additional_out = None, None
                if 'dataset_in' in self.dataset_batch:
                    additional_in = {}
                    for field in self.dataset_batch['dataset_in']:
                        additional_in[field] = self.dataset_batch['dataset_in'][field][0]
                        self.dataset_batch['dataset_in'][field] = self.dataset_batch['dataset_in'][field][1:]
                if 'dataset_out' in self.dataset_batch:
                    additional_out = {}
                    for field in self.dataset_batch['dataset_out']:
                        additional_out[field] = self.dataset_batch['dataset_out'][field][0]
                        self.dataset_batch['dataset_out'][field] = self.dataset_batch['dataset_out'][field][1:]
                self.environment.setDatasetSample(sample_in, sample_out, additional_in, additional_out)

            if animate:
                for current_step in range(self.simulations_per_step):
                    if current_step != self.simulations_per_step - 1:
                        self.environment.compute_essential_data = False
                        run(self.environment.step())
                    else:
                        self.environment.compute_essential_data = True
                        run(self.environment.step())

            if self.environment.checkSample() or not self.train:
                # Network's input
                if get_inputs:
                    inputs.append(self.environment.input)
                # Network's output
                if get_outputs:
                    outputs.append(self.environment.output)
                # Check if there is loss data
                if self.environment.loss_data is not None:
                    if 'loss' not in data_dict:
                        data_dict['loss'] = []
                    data_dict['loss'].append(self.environment.loss_data)
                # Check if there is additional input data fields
                if self.environment.additional_inputs != {}:
                    if 'dataset_in' not in data_dict:
                        data_dict['dataset_in'] = {}
                    for field in self.environment.additional_inputs:
                        if field not in data_dict['dataset_in']:
                            data_dict['dataset_in'][field] = []
                        data_dict['dataset_in'][field].append(self.environment.additional_inputs[field])
                # Check if there is additional output data fields
                if self.environment.additional_outputs != {}:
                    if 'dataset_out' not in data_dict:
                        data_dict['dataset_out'] = {}
                    for field in self.environment.additional_outputs:
                        if field not in data_dict['dataset_out']:
                            data_dict['dataset_out'][field] = []
                        data_dict['dataset_out'][field].append(self.environment.additional_outputs[field])
        # Convert network data
        training_data = {'input': array(inputs),
                         'output': array(outputs)}
        # Convert additional data
        for key in ['dataset_in', 'dataset_out']:
            if key in data_dict.keys():
                for field in data_dict[key]:
                    data_dict[key][field] = array(data_dict[key][field])
                training_data[key] = data_dict[key]
        # Convert loss data
        if 'loss' in data_dict.keys():
            training_data['loss'] = data_dict['loss']
        return training_data

    def updateVisualizer(self, visualization_data,):
        self.visualizer_manager.updateVisualizer(visualization_data)
        self.visualizer_manager.render()

    def dispatchBatch(self, batch):
        """
        Send samples from dataset to the Environments. Get back the training data.

        :param batch: Batch of samples.
        :return: Batch of training data.
        """
        if self.server:
            self.server.setDatasetBatch(batch)
        if self.environment:
            self.dataset_batch = batch
        return self.getData(animate=True)

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
