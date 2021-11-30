from os.path import join as osPathJoin
from os.path import isfile, isdir
from os import listdir, fstat, stat

from numpy import array, load, squeeze

from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX_Core.utils.pathUtils import getFirstCaller, createDir


class DatasetManager:

    def __init__(self,
                 dataset_config: BaseDatasetConfig,
                 data_manager=None,
                 session_name='default',
                 session_dir=None,
                 new_session=True,
                 train=True,
                 record_data=None):

        """
        DatasetManager handle all operations with input / output files. Allows to save and read tensors from files.

        :param BaseDatasetConfig dataset_config: Specialisation containing the parameters of the dataset manager
        :param DataManager data_manager: DataManager that handles the DatasetManager
        :param str session_name: Name of the newly created directory if session_dir is not defined
        :param str session_dir: Name of the directory in which to write all of the necessary data
        :param bool new_session: Define the creation of new directories to store data
        :param bool train: True if this session is a network training
        :param dict record_data: Format {\'in\': bool, \'out\': bool} save the tensor when bool is True
        """

        self.name = self.__class__.__name__
        self.data_manager = data_manager

        # Checking arguments
        if dataset_config is not None and not isinstance(dataset_config, BaseDatasetConfig):
            raise TypeError(f"[{self.name}] The dataset config must be a BaseDatasetConfig object.")
        if type(session_name) != str:
            raise TypeError(f"[{self.name}] The session name must be a str.")
        if session_dir is not None and type(session_dir) != str:
            raise TypeError(f"[{self.name}] The session directory must be a str.")
        if type(new_session) != bool:
            raise TypeError(f"[{self.name}] The 'new_network' argument must be a boolean.")
        if type(train) != bool:
            raise TypeError(f"[{self.name}] The 'train' argument must be a boolean.")
        if record_data is not None and type(record_data) != dict:
            raise TypeError(f"[{self.name}] The 'record_data' argument must be a dict.")
        elif record_data is not None:
            if type(record_data['input']) != bool or type(record_data['output']) != bool:
                raise TypeError(f"[{self.name}] The values of 'record_data' must be booleans.")

        # Create the dataset
        dataset_config = BaseDatasetConfig() if dataset_config is None else dataset_config
        self.dataset = dataset_config.createDataset()

        # Get dataset parameters
        self.max_size = self.dataset.max_size
        self.shuffle_dataset = dataset_config.shuffle_dataset
        self.record_data = record_data if record_data is not None else {'input': True, 'output': True}

        # Partition variables
        self.modes = {'Training': 0, 'Validation': 1, 'Running': 2}
        self.mode = self.modes['Training'] if train else self.modes['Running']
        self.last_loaded_dataset_mode = self.mode
        self.partitions_templates = (session_name + '_training_{}_{}.npy',
                                     session_name + '_validation_{}_{}.npy',
                                     session_name + '_running_{}_{}.npy')
        self.partitions_list_files = ('Training_partitions.txt',
                                      'Validation_partitions.txt',
                                      'Running_partitions.txt')
        self.fields = {'IN': ['input'], 'OUT': ['output']}

        self.list_partitions = {'input': [[], [], []] if self.record_data['input'] else None,
                                'output': [[], [], []] if self.record_data['output'] else None}
        self.idx_partitions = [0, 0, 0]
        self.current_partition_path = {'input': None, 'output': None}
        self.current_partition_file = {'input': None, 'output': None}

        self.saved = True
        self.first_add = True

        self.read_list_path = {field: [] for field in self.fields['IN'] + self.fields['OUT']}
        self.read_list_file, self.read_sizes, self.read_loaded = {}, {}, {}
        self.read_end_partitions = {field: [] for field in self.fields['IN'] + self.fields['OUT']}

        # Init dataset directories
        self.session_dir = session_dir if session_dir is not None else osPathJoin(getFirstCaller(), session_name)
        dataset_dir = dataset_config.dataset_dir
        self.new_session = new_session
        # Training
        if train:
            if new_session:
                if dataset_dir is None:  # New dataset
                    self.dataset_dir = createDir(dir_path=osPathJoin(self.session_dir, 'dataset/'),
                                                 dir_name='dataset')
                    self.createNewPartitions()
                else:  # Train from another session's dataset
                    if dataset_dir[-1] != "/":
                        dataset_dir += "/"
                    if dataset_dir[-8:] != "dataset/":
                        dataset_dir += "dataset/"
                    self.dataset_dir = dataset_dir
                    if dataset_config.input_shape is None or dataset_config.output_shape is None:
                        raise ValueError(f"[{self.name}] Data shapes must be set in DatasetConfig when loading an "
                                         f"existing dataset.")
                    self.dataset.init_data_size('input', dataset_config.input_shape)
                    self.dataset.init_data_size('output', dataset_config.output_shape)
                    self.loadDirectory()
            else:  # Train from this session's dataset
                self.dataset_dir = osPathJoin(self.session_dir, 'dataset/')
                if dataset_config.input_shape is None or dataset_config.output_shape is None:
                    raise ValueError(f"[{self.name}] Data shapes must be set in DatasetConfig when loading an "
                                     f"existing dataset.")
                self.dataset.init_data_size('input', dataset_config.input_shape)
                self.dataset.init_data_size('output', dataset_config.output_shape)
                self.loadDirectory()
        # Prediction
        else:
            self.dataset_dir = osPathJoin(self.session_dir, 'dataset/')
            self.createRunningPartitions()

    def getDataManager(self):
        """
        Return the Manager of the DataManager.

        :return: DataManager that handle The DatasetManager
        """
        return self.data_manager

    def createNewPartitions(self):
        """
        Generate a new partition (file of a certain maximum size). Input and output are generated
        independently if specified by record_data

        :return:
        """
        # Open the file containing the list of partitions files for the current mode
        print(f"New Partition: A new partition has been created with max size ~{float(self.max_size) / 1e9}Gb")
        file = osPathJoin(self.dataset_dir, self.partitions_list_files[self.mode])
        partitions_list_file = open(file, 'a')
        # Loop on all the registered data fields
        for side in self.fields:
            for field in self.fields[side]:
                if self.record_data[field]:
                    name = side if field in ['input', 'output'] else field
                    partition_name = self.partitions_templates[self.mode].format(name, self.idx_partitions[self.mode])
                    print(f"               {'In' if side == 'IN' else 'Out'}puts: {self.dataset_dir + partition_name}")
                    self.list_partitions[field][self.mode].append(partition_name)
                    partitions_list_file.write(partition_name + '\n')
                    self.current_partition_path[field] = self.dataset_dir + partition_name
                    self.current_partition_file[field] = open(self.current_partition_path[field], 'ab')

        self.idx_partitions[self.mode] += 1
        partitions_list_file.close()

    def register_new_field(self, side, field):
        """
        Add a new data field in the dataset.

        :param side: Either 'IN' side or 'OUT' side of the dataset
        :param field: Name of the new field
        :return:
        """
        # Register new field
        self.fields[side].append(field)
        self.list_partitions[field] = [[], [], []]
        self.record_data[field] = True
        # Open the file containing the list of partitions files for the current mode
        file = osPathJoin(self.dataset_dir, self.partitions_list_files[self.mode])
        partitions_list_file = open(file, 'a')
        # Create partition for the new field
        partition_name = self.partitions_templates[self.mode].format(field, self.idx_partitions[self.mode] - 1)
        self.list_partitions[field][self.mode].append(partition_name)
        partitions_list_file.write(partition_name + '\n')
        self.current_partition_path[field] = self.dataset_dir + partition_name
        self.current_partition_file[field] = open(self.current_partition_path[field], 'ab')
        partitions_list_file.close()

    def createRunningPartitions(self):
        """
        Run specific function. Handle partitions creation when not training.

        :return:
        """
        # 0. Check that the dataset repository is existing
        if not isdir(self.dataset_dir):
            raise Warning(f"[{self.name}]: The given path is not an existing directory.")
        # 1. Check whether if some running partitions
        running_partitions_file = [f for f in listdir(self.dataset_dir) if
                                   isfile(osPathJoin(self.dataset_dir, f)) and
                                   f.endswith('Running_partitions.txt')]
        # 1.1. No list file found, do a manual search for the partitions
        if running_partitions_file:
            print(f"[{self.name}] Listing file not found, searching for existing running partitions.")
            running_in_partitions = [f for f in listdir(self.dataset_dir) if
                                     isfile(osPathJoin(self.dataset_dir, f)) and f.endswith('.npy') and
                                     f.__contains__('running_IN')]
            running_out_partitions = [f for f in listdir(self.dataset_dir) if
                                      isfile(osPathJoin(self.dataset_dir, f)) and f.endswith('.npy') and
                                      f.__contains__('running_OUT')]
        # 1.2. Normally there is a single list of partitions per mode
        elif len(running_partitions_file) != 1:
            raise ValueError(f"[{self.name}] It appears that several running partition lists have been found.")
        # 1.3. Simply get the partitions from the list file
        else:
            reader = open(osPathJoin(self.dataset_dir, running_partitions_file[0]))
            running_partitions_list = reader.read().splitlines()
            running_in_partitions = [f for f in running_partitions_list if
                                     isfile(osPathJoin(self.dataset_dir, f)) and f.endswith('.npy') and
                                     f.__contains__('running_IN')]
            running_out_partitions = [f for f in running_partitions_list if
                                      isfile(osPathJoin(self.dataset_dir, f)) and f.endswith('.npy') and
                                      f.__contains__('running_OUT')]
        # 2. Create the appropriate partitions
        nb_running_partitions = max(len(running_in_partitions), len(running_out_partitions))
        self.idx_partitions[self.mode] = nb_running_partitions
        self.createNewPartitions()

    def loadDirectory(self):
        """
        Load the desired directory. Try to find partition list and upload it.
        No data loading here.

        :return:
        """
        # Load a directory according to the distribution given by the lists
        print(f"[{self.name}] Loading directory: Read dataset from {self.dataset_dir}")
        if not isdir(self.dataset_dir):
            raise Warning(f"[{self.name}] Loading directory: The given path is not an existing directory")
        # Look for file which ends with 'mode_partitions.txt'
        for mode in ['Training', 'Validation', 'Running']:
            partitions_list_file = [f for f in listdir(self.dataset_dir) if
                                    isfile(osPathJoin(self.dataset_dir, f)) and
                                    f.endswith(mode[1:] + '_partitions.txt')]
            # If there is no such files then proceed to load any dataset found as in/out
            if not partitions_list_file:
                print(f"Loading directory: Partitions list not found for {mode} mode, will consider any .npy file as "
                      "input/output.")
                partitions_list = [f for f in listdir(self.dataset_dir) if
                                   isfile(osPathJoin(self.dataset_dir, f)) and f.endswith(".npy") and
                                   f.__contains__(mode)]
            # If partitions_list.txt found then proceed to load the specific dataset as input/output
            else:
                reader = open(self.dataset_dir + partitions_list_file[0])
                partitions_list = reader.read().splitlines()
                reader.close()
            # Split partitions in sides
            in_partitions = sorted([file for file in partitions_list if file.__contains__('_IN_')])
            out_partitions = sorted([file for file in partitions_list if file.__contains__('_OUT_')])
            # Classify by field
            for side, side_partitions, network_field in zip(['IN', 'OUT'], [in_partitions, out_partitions],
                                                            ['input', 'output']):
                for partition in side_partitions:
                    split_name = partition.split('_')
                    indicators = split_name[split_name.index(side) + 1:]
                    # One indicator: index of a partition for a network input
                    if len(indicators) == 1:
                        self.list_partitions[network_field][self.modes[mode]].append(partition)
                    # More than one indicator: field name + index for an additional input
                    else:
                        field_name = side
                        for ind in indicators[:-1]:
                            field_name += '_' + ind
                        self.register_new_field(side=side, field=field_name)
                        self.list_partitions[field_name][self.modes[mode]].append(partition)
            # Check that the number of partitions is the same for each field
            number_of_partitions = len(self.list_partitions[self.fields['IN'][0]][self.modes[mode]])
            for field in self.fields['IN'] + self.fields['OUT']:
                if len(self.list_partitions[field][self.modes[mode]]) != number_of_partitions:
                    raise ValueError(f"[{self.name}] The number of partitions is different for {field} with "
                                     f"{len(self.list_partitions[field][self.modes[mode]])} partitions found.")
        self.loadPartitions(force_partition_reload=True)

    def requireEnvironment(self):
        """
        Called while training to check if each inputs as an output, otherwise need an environment to compute it

        :return: True if need to compute a new sample
        """
        # self.new_session or
        return self.new_session or \
               len(self.list_partitions['input'][0]) > len(self.list_partitions['output'][0]) or \
               len(self.list_partitions['input'][1]) > len(self.list_partitions['output'][1]) or \
               len(self.list_partitions['input'][2]) > len(self.list_partitions['output'][2])

    def addData(self, data):
        """
        Push the data in the dataset. If max size is reached generate a new partition and write into it.

        :param dict data: Format {'input':numpy.ndarray, 'output':numpy.ndarray}  contain in 'input' input tensors and
        in 'output' output tensors.

        :return:
        """
        self.saved = False
        # 1. Adding network data to dataset
        for field in ['input', 'output']:
            if self.record_data[field]:
                self.dataset.add(field, data[field], self.current_partition_file[field])
        # 2. Add additional dataset
        for side, key in zip(['IN', 'OUT'], ['dataset_in', 'dataset_out']):
            additional_data = {}
            for field in data[key].keys():
                additional_data[side + '_' + field] = data[key][field]
            if additional_data != {}:
                self.add_additionalData(side, additional_data)
        self.first_add = False
        # 3. Check the size of the dataset (input + output) (if only input, consider the virtual size of the output)
        if self.dataset.memory_size() > self.max_size:
            self.saveData()
            self.createNewPartitions()
            self.dataset.reset()

    def add_additionalData(self, side, additional_data):
        """
        Push an additional data in the dataset.

        :param side: Either 'IN' side or 'OUT' side of the dataset
        :param additional_data: Batch of additional dataset
        :return:
        """
        # If partitions exists other than in / out, check that a sample is given for each additional partition
        if len(self.fields[side][1:]) > 0 and not self.first_add:
            for field in self.fields[side][1:]:
                if field not in additional_data:
                    raise ValueError(f"[{self.name}] No data received for the additional dataset field {field}.")
        # Add data for each field
        for field in additional_data:
            # First time, key does not exists
            if field not in self.fields['IN'][1:] + self.fields['OUT'][1:]:
                self.register_new_field(side, field)
            # Add data to field
            self.dataset.add(field, additional_data[field], self.current_partition_file[field])

    def saveData(self):
        """
        Close all open files

        :return:
        """
        self.saved = True
        for field in self.current_partition_file.keys():
            self.current_partition_file[field].close()

    def setMode(self, mode):
        """
        Set the DatasetManager working mode.

        :param int mode: Recommended to use datasetManager.modes['name_of_desired_mode'] in order to correctly set up
        the DatasetManager

        :return:
        """
        # Nothing has to be done if you do not change mode
        if mode == self.mode:
            return
        if self.mode == self.modes['Running']:
            print(f"[{self.name}] It's not possible to switch dataset mode while running.")
        else:
            # Save dataset before changing mode
            if not self.saved:
                self.saveData()
            self.mode = mode
            self.dataset.reset()
            # Create or load partition for the new mode
            if self.idx_partitions[self.mode] == 0:
                print(f"[{self.name}] Change to {self.mode} mode, create a new partition")
                self.createNewPartitions()
            else:
                print(f"[{self.name}] Change to {self.mode} mode, load last partition")
                self.loadLastPartitions()

    def loadLastPartitions(self):
        """
        Load the last partition is the partition list.

        :return:
        """
        # Load last created partition for each data field
        for field in self.fields['IN'] + self.fields['OUT']:
            # Get the path of he partition
            self.current_partition_path[field] = self.dataset_dir + self.list_partitions[field][self.mode][-1]
            with open(self.current_partition_path[field], 'rb') as file:
                size = fstat(file.fileno()).st_size
                while self.dataset.memory_size(field) < size:
                    size -= 128  # Each array takes 128 extra bytes in memory
                    data = load(file)
                    self.dataset.add(field, array([data]))

    def getData(self, get_inputs, get_outputs, batch_size=1, batched=True, force_partition_reload=False):
        """
        Fetch tensors from the dataset or reload partitions if dataset is empty or specified.

        :param bool get_inputs: If True fill the data['input'] field
        :param bool get_outputs: If True fill the data['output'] field
        :param int batch_size: Size of a batch
        :param bool batched: Add an empty dimension before [4,100] -> [0,4,100]
        :param bool force_partition_reload: If True force reload of partition

        :return: dict of format {'input':numpy.ndarray, 'output':numpy.ndarray} filled with desired data
        """
        # Check if at least input field is loaded
        if self.current_partition_path['input'] is None or self.dataset.current_sample >= len(
                self.dataset.data['input']):
            if not force_partition_reload:
                return None
            self.loadPartitions()
            if self.shuffle_dataset:
                self.dataset.shuffle()
            self.dataset.current_sample = 0
        idx = self.dataset.current_sample
        data = {}
        # Get batch for each input / output field
        fields = self.fields['IN'][:] if get_inputs else []
        fields += self.fields['OUT'][:] if get_outputs else []
        # Get data for each additional field
        for field in fields:
            if field in ['input', 'output']:
                data[field] = self.dataset.get(field, idx, idx + batch_size)
                if not batched:
                    data[field] = squeeze(data[field], axis=0)
            else:
                side = 'dataset_in' if field[:3] == 'IN_' else 'dataset_out'
                if side not in data.keys():
                    data[side] = {}
                user_field = field[3:] if side == 'dataset_in' else field[4:]
                data[side][user_field] = self.dataset.get(field, idx, idx + batch_size)
                if not batched:
                    data[side][user_field] = squeeze(data[side][user_field], axis=0)
        # Index dataset
        self.dataset.current_sample += batch_size
        return data

    def getNextBatch(self, batch_size):
        """
        :param int batch_size: Size of the batch
        :return: dict of format {'input': numpy.ndarray, 'output': numpy.ndarray} filled with a batch of data
        """
        return self.getData(get_inputs=True, get_outputs=True, batch_size=batch_size, batched=True)

    def getNextSample(self, batched=True):
        """
        :return: dict of format {'input': numpy.ndarray, 'output': numpy.ndarray} filled with a sample of data
        """
        return self.getData(get_inputs=True, get_outputs=True, batched=batched)

    def getNextInput(self, batched=False):
        """
        :return: dict of format {'input': numpy.ndarray, 'output': numpy.ndarray} where only the input field is filled
        """
        return self.getData(get_inputs=True, get_outputs=False, batched=batched)

    def getNextOutput(self, batched=False):
        """
        :return: dict of format {'input': numpy.ndarray, 'output': numpy.ndarray} where only the output field is filled
        """
        return self.getData(get_inputs=False, get_outputs=True, batched=batched)

    def loadPartitions(self, force_partition_reload=False):
        """
        Load partitions as specified in the class initialisation. At the end of the function the dataset hopefully
        is non empty.

        :param bool force_partition_reload: If True force reload of partition
        :return:
        """
        # If there is only one partition for the current mode for input field at least, don't need to reload it
        if self.last_loaded_dataset_mode == self.mode and len(self.list_partitions['input'][self.mode]) == 1\
                and not force_partition_reload:
            print("LOAD PARTITION SKIP")
            return
        # Otherwise reload partitions
        self.dataset.reset()
        # Testing mode
        if self.mode == self.modes['Validation']:
            # Check if some partitions exist for this mode for input field at least
            if len(self.list_partitions['input'][self.mode]) == 0:
                raise ValueError(f"[{self.name}] No partitions to read for testing mode.")
            elif len(self.list_partitions['input'][self.mode]) == 1:
                self.loadLastPartitions()
            else:
                self.loadMultiplePartitions([self.mode])
        # Training mode, loadPartition not called in running mode
        else:
            # Mixed dataset
            if len(self.list_partitions['input'][self.modes['Running']]) > 0:
                self.loadMultiplePartitions([self.modes['Training'], self.modes['Running']])
            else:
                if len(self.list_partitions['input'][self.mode]) == 0:
                    raise ValueError("[{}] No partitions to read for training mode.")
                elif len(self.list_partitions['input'][self.mode]) == 1:
                    self.loadLastPartitions()
                else:
                    self.loadMultiplePartitions([self.mode])

    def loadMultiplePartitions(self, modes):
        """
        Specialisation of the loadPartitions function. It can load a list of partitions

        :param list modes: Recommended to use datasetManager.modes['name_of_desired_mode'] in order to correctly load
        the dataset
        :return:
        """
        if self.endReadPartitions():
            self.read_list_path = {field: [] for field in self.fields['IN'] + self.fields['OUT']}
            self.read_list_file, self.read_sizes, self.read_loaded = {}, {}, {}
            self.read_end_partitions = {field: [] for field in self.fields['IN'] + self.fields['OUT']}
            # Open all partitions, get sizes before to load in the dataset
            for field in self.fields['IN'] + self.fields['OUT']:
                for mode in modes:
                    self.read_list_path[field] += [self.dataset_dir + partition for partition in self.list_partitions[field][mode]]
                self.read_list_file[field] = [open(path, 'rb') for path in self.read_list_path[field]]
                self.read_sizes[field] = [stat(file.fileno()).st_size for file in self.read_list_file[field]]
                self.read_loaded[field] = [0.] * len(self.read_list_file[field])
                self.read_end_partitions[field] = [False for _ in self.read_list_path[field]]
        self.readMultiplePartitions()
            
    def readMultiplePartitions(self):
        """
        Read data in a list of partitions.

        :return:
        """
        idx_file = 0
        # Load fields until dataset is full
        while self.dataset.memory_size() < self.max_size and not self.endReadPartitions():
            for field in self.read_list_path.keys():
                if not self.read_end_partitions[field][idx_file]:
                    data = load(self.read_list_file[field][idx_file])
                    self.read_sizes[field][idx_file] -= 128
                    self.read_loaded[field][idx_file] += data.nbytes
                    if self.read_loaded[field][idx_file] >= self.read_sizes[field][idx_file]:
                        self.read_end_partitions[field][idx_file] = True
                    self.dataset.add(field, array([data]))
            idx_file = (idx_file + 1) % len(self.read_list_path[field])
        self.current_partition_path['input'] = self.read_list_path[field][idx_file]
        self.dataset.current_sample = 0

    def endReadPartitions(self):
        """
        Check if all reading partitions are done.

        :return:
        """
        res = True
        for field in self.read_end_partitions:
            for check in self.read_end_partitions[field]:
                res = check and res
        return res

    def close(self):
        """
        Launch the close procedure of the dataset manager

        :return:
        """
        if not self.saved:
            self.saveData()

    def __str__(self):
        """
        :return: A string containing valuable information about the DatasetManager
        """
        description = "\n"
        description += f"# {self.name}\n"
        description += f"    Dataset Repository: {self.dataset_dir}\n"
        description += f"    Partitions size: {self.max_size * 1e-9} Go\n"
        description += f"    Managed objects: Dataset: {self.dataset.name}\n"
        description += str(self.dataset)
        return description
