from os.path import join as osPathJoin
from os.path import isfile, isdir
from os import listdir, fstat, stat

from numpy import array, load, squeeze

from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX_Core.utils.pathUtils import getFirstCaller, createDir


class DatasetManager:

    def __init__(self, dataset_config: BaseDatasetConfig, data_manager=None, session_name='default', session_dir=None, new_session=True,
                 train=True, record_data=None):

        """
        DatasetManager handle all operations with input / output files. Allows to save and read tensors from files.

        :param BaseDatasetConfig dataset_config: Specialisation containing the parameters of the dataset manager
        :param DataManager data_manager: DataManager that handles the DatasetManager
        :param str session_name: Name of the newly created directory if session_dir is not defined
        :param str session_dir: Name of the directory in which to write all of the neccesary data
        :param bool new_session: Define the creation of new directories to store data
        :param bool training: True if this session is a network training
        :param dict record_data: Format {\'in\': bool, \'out\': bool} save the tensor when bool is True
        """
        self.name = self.__class__.__name__
        self.data_manager = data_manager
        # Checking arguments
        if dataset_config is not None and not isinstance(dataset_config, BaseDatasetConfig):
            raise TypeError("[DATASETMANAGER] The dataset config must be a BaseDatasetConfig object.")
        if type(session_name) != str:
            raise TypeError("[DATASETMANAGER] The session name must be a str.")
        if session_dir is not None and type(session_dir) != str:
            raise TypeError("[DATASETMANAGER] The session directory must be a str.")
        if type(new_session) != bool:
            raise TypeError("[DATASETMANAGER] The 'new_network' argument must be a boolean.")
        if type(train) != bool:
            raise TypeError("[DATASETMANAGER] The 'train' argument must be a boolean.")
        if record_data is not None and type(record_data) != dict:
            raise TypeError("[DATASETMANAGER] The 'record_data' argument must be a dict.")
        elif record_data is not None:
            if type(record_data['in']) != bool or type(record_data['out']) != bool:
                raise TypeError("[DATASETMANAGER] The values of 'record_data' must be booleans.")

        # Create the dataset
        dataset_config = BaseDatasetConfig() if dataset_config is None else dataset_config
        self.dataset = dataset_config.createDataset()

        # Get dataset parameters
        self.max_size = self.dataset.max_size
        self.shuffle_dataset = dataset_config.shuffle_dataset
        self.record_data = record_data if record_data is not None else {'in': True, 'out': True}

        # Partition variables
        self.modes = {'Training': 0, 'Validation': 1, 'Running': 2}
        self.mode = self.modes['Training'] if train else self.modes['Running']
        self.partitions_templates = (session_name + '_training_{}_{}.npy',
                                     session_name + '_validation_{}_{}.npy',
                                     session_name + '_running_{}_{}.npy')
        self.partitions_list_files = ('Training_partitions.txt',
                                      'Validation_partitions.txt',
                                      'Running_partitions.txt')
        self.list_in_partitions = [[], [], []] if self.record_data['in'] else None
        self.list_out_partitions = [[], [], []] if self.record_data['out'] else None
        self.idx_partitions = [0, 0, 0]
        self.current_in_partition_name, self.current_out_partition_name = None, None
        self.current_in_partition, self.current_out_partition = None, None
        self.saved = True

        # Init dataset directories
        self.session_dir = session_dir if session_dir is not None else osPathJoin(getFirstCaller(),
                                                                                    session_name)
        dataset_dir = dataset_config.dataset_dir
        self.new_session = new_session
        # Training
        if train:
            if new_session:
                if dataset_dir is None:  # New dataset
                    self.dataset_dir = createDir(dirname=osPathJoin(self.session_dir, 'dataset/'),
                                                           check_existing='dataset')
                    self.createNewPartitions()
                else:  # Train from another session's dataset
                    if dataset_dir[-1] != "/":
                        dataset_dir += "/"
                    if dataset_dir[-8:] != "dataset/":
                        dataset_dir += "dataset/"
                    self.dataset_dir = dataset_dir
                    self.loadDirectory()
            else:  # Train from this session's dataset
                self.dataset_dir = osPathJoin(self.session_dir, 'dataset/')
                self.loadDirectory()

        # Prediction
        else:
            self.dataset_dir = osPathJoin(self.session_dir, 'dataset/')
            self.createRunningPartitions()

    def getDataManager(self):
        """

        :return: DataManager that handle The DatasetManager
        """
        return self.data_manager

    def createNewPartitions(self):
        """
        Generate a new partition (file of a certain maximum size). Input and output are generated
        independently if specified by record_data

        :return:
        """
        # Create in and out partitions
        print("New Partition: A new partition has been created with max size ~{}Gb".format(float(self.max_size) / 1e9))
        file = osPathJoin(self.dataset_dir, self.partitions_list_files[self.mode])
        partitions_list_file = open(file, 'a')

        if self.record_data['in']:
            current_part_in = self.partitions_templates[self.mode].format('IN', self.idx_partitions[self.mode])
            print("               Inputs: {}".format(self.dataset_dir + current_part_in))
            self.list_in_partitions[self.mode].append(current_part_in)
            partitions_list_file.write(current_part_in + '\n')
            self.current_in_partition_name = self.dataset_dir + current_part_in
            self.current_in_partition = open(self.current_in_partition_name, 'ab')

        if self.record_data['out']:
            current_part_out = self.partitions_templates[self.mode].format('OUT', self.idx_partitions[self.mode])
            print("               Outputs: {}".format(self.dataset_dir + current_part_out))
            self.list_out_partitions[self.mode].append(current_part_out)
            partitions_list_file.write(current_part_out + '\n')
            self.current_out_partition_name = self.dataset_dir + current_part_out
            self.current_out_partition = open(self.current_out_partition_name, 'ab')

        self.idx_partitions[self.mode] += 1
        partitions_list_file.close()

    def createRunningPartitions(self):
        """
        Run specific function. Handle partitions creation when not training.

        :return:
        """
        # 0. Check that the dataset repository is existing
        if not isdir(self.dataset_dir):
            raise Warning("[{}]: The given path is not an existing directory.".format(self.name))
        # 1. Check whether if some running partitions
        running_partitions_file = [f for f in listdir(self.dataset_dir) if
                                   isfile(osPathJoin(self.dataset_dir, f)) and
                                   f.endswith('Running_partitions.txt')]
        # 1.1. No list file found, do a manual search for the partitions
        if running_partitions_file:
            print("[{}] Listing file not found, searching for existing running partitions.".format(self.name))
            running_in_partitions = [f for f in listdir(self.dataset_dir) if
                                     isfile(osPathJoin(self.dataset_dir, f)) and f.endswith('.npy') and
                                     f.__contains__('running_IN')]
            running_out_partitions = [f for f in listdir(self.dataset_dir) if
                                      isfile(osPathJoin(self.dataset_dir, f)) and f.endswith('.npy') and
                                      f.__contains__('running_OUT')]
        # 1.2. Normally there is a single list of partitions per mode
        elif len(running_partitions_file) != 1:
            raise ValueError("[{}] It appears that several running partition lists have been found.")
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
        print("Loading directory: Read dataset from {}".format(self.dataset_dir))
        if not isdir(self.dataset_dir):
            raise Warning("Loading directory: The given path is not an existing directory")
        # Look for file which ends with 'mode_partitions.txt'
        for mode in ['Training', 'Validation', 'Running']:
            partitions_list_file = [f for f in listdir(self.dataset_dir) if
                                    isfile(osPathJoin(self.dataset_dir, f)) and
                                    f.endswith(mode[1:] + '_partitions.txt')]
            # If there is no such files then proceed to load any dataset found as in/out
            if not partitions_list_file:
                print("Loading directory: Partitions list not found for {} mode, will consider any .npy file as "
                      "input/output.".format(mode))
                partitions_list = [f for f in listdir(self.dataset_dir) if
                                   isfile(osPathJoin(self.dataset_dir, f)) and f.endswith(".npy") and
                                   f.__contains__(mode)]
            # If partitions_list.txt found then proceed to load the specific dataset as input/output
            else:
                reader = open(self.dataset_dir + partitions_list_file[0])
                partitions_list = reader.read().splitlines()
                reader.close()
            # Add partitions to lists
            nb_parts = len(partitions_list)
            partitions_in = sorted([f for f in partitions_list if f.__contains__('IN')])
            partitions_out = sorted([f for f in partitions_list if f.__contains__('OUT')])
            if nb_parts != len(partitions_in) + len(partitions_out):
                raise ValueError("[{}] The number of partitions is ambiguous.".format(self.name))
            self.list_in_partitions[self.modes[mode]] = partitions_in
            self.list_out_partitions[self.modes[mode]] = partitions_out

    def requireEnvironment(self):
        """
        Called while training to check if each inputs as an output, otherwise need an environment to compute it

        :return: True if need to compute a new sample
        """
        # self.new_session or
        return self.new_session or len(self.list_in_partitions[0]) > len(self.list_out_partitions[0]) or \
               len(self.list_in_partitions[1]) > len(self.list_out_partitions[1]) or \
               len(self.list_in_partitions[2]) > len(self.list_out_partitions[2])

    def addData(self, data):
        """
        Push the data in the dataset. If max size is reached generate a new partition and write into it.

        :param dict data: Format {'in':numpy.ndarray, 'out':numpy.ndarray}  contain in 'in' input tensors and in
        'out' output tensors.

        :return:
        """
        self.saved = False
        # 1. Adding data to dataset
        if self.record_data['in']:
            self.dataset.add('in', data['in'], self.current_in_partition)
        if self.record_data['out']:
            self.dataset.add('out', data['out'], self.current_out_partition)
        # 2. Check the size of the dataset (input + output) (if only input, consider the virtual size of the output)
        max_size = self.max_size if self.record_data['in'] and self.record_data['out'] else self.max_size / 2
        if self.dataset.memory_size() > max_size:
            self.saveData()
            self.createNewPartitions()
            self.dataset.reset()

    def saveData(self):
        """
        Close all open files

        :return:
        """

        self.saved = True
        self.current_in_partition.close()
        self.current_out_partition.close()

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
            print("[{}] It's not possible to switch dataset mode while running.".format(self.name))
        else:
            # Save dataset before changing mode
            if not self.saved:
                self.saveData()
            self.mode = mode
            self.dataset.reset()
            # Create or load partition for the new mode
            if self.idx_partitions[self.mode] == 0:
                print("Change to {} mode, create a new partition".format(self.mode))
                self.createNewPartitions()
            else:
                print("Change to {} mode, load last partition".format(self.mode))
                self.loadLastPartitions()

    def loadLastPartitions(self):
        """
        Load the last partition is the partion list

        :return:
        """
        # Input
        self.current_in_partition_name = self.dataset_dir + self.list_in_partitions[self.mode][-1]
        with open(self.current_in_partition_name, 'rb') as in_file:
            in_size = fstat(in_file.fileno()).st_size
            while self.dataset.data_in.nbytes < in_size:
                in_size -= 128  # Each array takes 128 extra bytes in memory
                data_in = load(in_file)
                self.dataset.load('in', data_in)
        # Output
        self.current_out_partition_name = self.dataset_dir + self.list_out_partitions[self.mode][-1]
        with open(self.current_out_partition_name, 'rb') as out_file:
            out_size = fstat(out_file.fileno()).st_size
            while self.dataset.data_out.nbytes < out_size:
                out_size -= 128  # Each array takes 128 extra bytes in memory
                data_out = load(out_file)
                self.dataset.load('out', data_out)

    def getData(self, get_inputs, get_outputs, batch_size=1, batched=True, force_partition_reload=False):
        """
        Fetch tensors from the dataset or reload partitions if dataset is empty or specified.

        :param bool get_inputs: If True fill the data['in'] field
        :param bool get_outputs: If True fill the data['out'] field
        :param int batch_size: Size of a batch
        :param bool batched: Add an empty dimension before [4,100] -> [0,4,100]
        :param bool force_partition_reload: If True force reload of partition

        :return: dict of format {'in':numpy.ndarray, 'out':numpy.ndarray} filled with desired data
        """
        if self.current_in_partition_name is None or self.dataset.current_sample >= len(self.dataset.data_in):
            if not force_partition_reload:
                return None
            self.loadPartitions()
            if self.shuffle_dataset:
                self.dataset.shuffle()
            self.dataset.current_sample = 0
        idx = self.dataset.current_sample
        data = {'in': array([]), 'out': array([])}
        if get_inputs:
            if batched:
                data['in'] = self.dataset.data_in[idx: idx + batch_size].reshape((-1, *self.dataset.in_shape))
            else:
                data['in'] = squeeze(self.dataset.data_in[idx: idx + batch_size], axis=0)
        if get_outputs:
            if batched:
                data['out'] = self.dataset.data_out[idx: idx + batch_size].reshape((-1, *self.dataset.out_shape))
            else:
                data['out'] = squeeze(self.dataset.data_out[idx: idx + batch_size], axis=0)
        self.dataset.current_sample += batch_size
        return data

    def getNextBatch(self, batch_size):
        """
        :param int batch_size: Size of the batch
        :return: dict of format {'in': numpy.ndarray, 'out': numpy.ndarray} filled with a batch of data
        """
        return self.getData(get_inputs=True, get_outputs=True, batch_size=batch_size, batched=True)

    def getNextSample(self, batched=True):
        """
        :return: dict of format {'in': numpy.ndarray, 'out': numpy.ndarray} filled with a sample of data
        """
        return self.getData(get_inputs=True, get_outputs=True, batched=batched)

    def getNextInput(self, batched=False):
        """
        :return: dict of format {'in': numpy.ndarray, 'out': numpy.ndarray} where only the input field is filled
        """
        return self.getData(get_inputs=True, get_outputs=False, batched=batched)

    def getNextOutput(self, batched=False):
        """
        :return: dict of format {'in': numpy.ndarray, 'out': numpy.ndarray} where only the output field is filled
        """
        return self.getData(get_inputs=False, get_outputs=True, batched=batched)

    def loadPartitions(self):
        """
        Load partitions as specified in the class initialisation. At the end of the function the dataset hopefully
        is non empty.

        :return:
        """
        self.dataset.reset()
        # Testing mode
        if self.mode == self.modes['Validation']:
            if len(self.list_in_partitions[self.mode]) == 0:
                raise ValueError("[{}] No partitions to read for testing mode.")
            elif len(self.list_in_partitions[self.mode]) == 1:
                self.loadLastPartitions()
            else:
                self.loadMultiplePartitions([self.mode])
        # Training mode, loadPartition not called in running mode
        else:
            # Mixed dataset
            if len(self.list_in_partitions[self.modes['Running']]) > 0:
                self.loadMultiplePartitions([self.modes['Training'], self.modes['Running']])
            else:
                if len(self.list_in_partitions[self.mode]) == 0:
                    raise ValueError("[{}] No partitions to read for training mode.")
                elif len(self.list_in_partitions[self.mode]) == 1:
                    self.loadLastPartitions()
                else:
                    self.loadMultiplePartitions([self.mode])

    def loadMultiplePartitions(self, modes):
        """
        Specialisation of the loadPartitions function. It can load a list of partitions
        :param int modes: Recommended to use datasetManager.modes['name_of_desired_mode'] in order to correctly load
        the dataset

        :return:
        """
        in_filenames, out_filenames = [], []
        for mode in modes:
            in_filenames += [self.dataset_dir + partition for partition in self.list_in_partitions[mode]]
            out_filenames += [self.dataset_dir + partition for partition in self.list_out_partitions[mode]]
        in_files = [open(filename, 'rb') for filename in in_filenames]
        out_files = [open(filename, 'rb') for filename in out_filenames]
        in_sizes = [stat(in_file.fileno()).st_size for in_file in in_files]
        out_sizes = [stat(out_file.fileno()).st_size for out_file in out_files]
        in_loaded = [0.] * len(in_sizes)
        out_loaded = [0.] * len(out_sizes)
        end_partition = False
        idx_file = 0
        while self.dataset.memory_size() < self.max_size:
            in_sizes[idx_file] -= 128
            data_in = load(in_files[idx_file])
            in_loaded[idx_file] += data_in.nbytes
            self.dataset.load('in', data_in)
            if in_loaded[idx_file] >= in_sizes[idx_file]:
                break

            try:
                out_sizes[idx_file] -= 128
                data_out = load(out_files[idx_file])
                out_loaded[idx_file] += data_out.nbytes
                self.dataset.load('out', data_out)
                if out_loaded[idx_file] >= out_sizes[idx_file]:
                    break
            except:
                pass

            idx_file = (idx_file + 1) % len(in_files)

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
