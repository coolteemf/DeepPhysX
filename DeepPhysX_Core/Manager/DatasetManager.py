import os
import numpy as np

from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
import DeepPhysX_Core.utils.pathUtils as pathUtils


class DatasetManager:

    def __init__(self, dataset_config=None, session_name='default', session_dir=None, new_session=True,
                 train=True, record_data=None):

        self.name = self.__class__.__name__

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
        self.session_dir = session_dir if session_dir is not None else os.path.join(pathUtils.getFirstCaller(),
                                                                                    session_name)
        dataset_dir = dataset_config.dataset_dir
        self.new_session = new_session
        # Training
        if train:
            if new_session:
                if dataset_dir is None:  # New dataset
                    self.dataset_dir = pathUtils.createDir(dirname=os.path.join(self.session_dir, 'dataset/'),
                                                           check_existing='dataset')
                    self.createNewPartitions()
                else:  # Train from another session's dataset
                    if dataset_dir[-1] != "/":
                        dataset_dir += "/"
                    if dataset_dir[-8:] != "dataset/":
                        dataset_dir += "dataset/"
                    self.dataset_dir = dataset_dir  # pathUtils.copyDir(src_dir=dataset_dir,
                                                         # dest_parent_dir=self.session_dir,
                                                         # dest_dir='dataset')
                    print(f"{self.dataset_dir=}")
                    self.loadDirectory()
            else:  # Train from this session's dataset
                self.dataset_dir = os.path.join(self.session_dir, 'dataset/')
                self.loadDirectory()

        # Prediction
        else:
            self.dataset_dir = os.path.join(self.session_dir, 'dataset/')
            self.createRunningPartitions()

        # Description
        self.description = ""

    def createNewPartitions(self):
        # Create in and out partitions
        print("New Partition: A new partition has been created with max size ~{}Gb".format(float(self.max_size) / 1e9))
        file = os.path.join(self.dataset_dir, self.partitions_list_files[self.mode])
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
        # 0. Check that the dataset repository is existing
        if not os.path.isdir(self.dataset_dir):
            raise Warning("[{}]: The given path is not an existing directory.".format(self.name))
        # 1. Check whether if some running partitions
        running_partitions_file = [f for f in os.listdir(self.dataset_dir) if
                                   os.path.isfile(os.path.join(self.dataset_dir, f)) and
                                   f.endswith('Running_partitions.txt')]
        # 1.1. No list file found, do a manual search for the partitions
        if len(running_partitions_file) == 0:
            print("[{}] Listing file not found, searching for existing running partitions.".format(self.name))
            running_in_partitions = [f for f in os.listdir(self.dataset_dir) if
                                     os.path.isfile(os.path.join(self.dataset_dir, f)) and f.endswith('.npy') and
                                     f.__contains__('running_IN')]
            running_out_partitions = [f for f in os.listdir(self.dataset_dir) if
                                      os.path.isfile(os.path.join(self.dataset_dir, f)) and f.endswith('.npy') and
                                      f.__contains__('running_OUT')]
        # 1.2. Normally there is a single list of partitions per mode
        elif len(running_partitions_file) != 1:
            raise ValueError("[{}] It appears that several running partition lists have been found.")
        # 1.3. Simply get the partitions from the list file
        else:
            reader = open(os.path.join(self.dataset_dir, running_partitions_file[0]))
            running_partitions_list = reader.read().splitlines()
            running_in_partitions = [f for f in running_partitions_list if
                                     os.path.isfile(os.path.join(self.dataset_dir, f)) and f.endswith('.npy') and
                                     f.__contains__('running_IN')]
            running_out_partitions = [f for f in running_partitions_list if
                                      os.path.isfile(os.path.join(self.dataset_dir, f)) and f.endswith('.npy') and
                                      f.__contains__('running_OUT')]
        # 2. Create the appropriate partitions
        nb_running_partitions = max(len(running_in_partitions), len(running_out_partitions))
        self.idx_partitions[self.mode] = nb_running_partitions
        self.createNewPartitions()

    def loadDirectory(self):
        # Load a directory according to the distribution given by the lists
        print("Loading directory: Read dataset from {}".format(self.dataset_dir))
        if not os.path.isdir(self.dataset_dir):
            raise Warning("Loading directory: The given path is not an existing directory")
        # Look for file which ends with 'mode_partitions.txt'
        for mode in ['Training', 'Validation', 'Running']:
            partitions_list_file = [f for f in os.listdir(self.dataset_dir) if
                                    os.path.isfile(os.path.join(self.dataset_dir, f)) and
                                    f.endswith(mode[1:] + '_partitions.txt')]
            # If there is no such files then proceed to load any dataset found as in/out
            if len(partitions_list_file) == 0:
                print("Loading directory: Partitions list not found for {} mode, will consider any .npy file as "
                      "input/output.".format(mode))
                partitions_list = [f for f in os.listdir(self.dataset_dir) if
                                   os.path.isfile(os.path.join(self.dataset_dir, f)) and f.endswith(".npy") and
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
        # Called while training to check if each inputs as an output, otherwise need an environment to compute it
        if self.new_session:
            return True
        for mode in range(3):
            if len(self.list_in_partitions[mode]) > len(self.list_out_partitions[mode]):
                return True
        return False

    def addData(self, data):
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
        self.saved = True
        self.current_in_partition.close()
        self.current_out_partition.close()

    def setMode(self, mode):
        # Nothing has to be done if you do not change mode
        if mode == self.mode:
            return
        if self.mode == self.modes['Running']:
            print("[{}] It's not possible to switch dataset mode while running.".format(self.name))
            return
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
        # Input
        self.current_in_partition_name = self.dataset_dir + self.list_in_partitions[self.mode][-1]
        with open(self.current_in_partition_name, 'rb') as in_file:
            in_size = os.fstat(in_file.fileno()).st_size
            while self.dataset.data_in.nbytes < in_size:
                in_size -= 128  # Each array takes 128 extra bytes in memory
                data_in = np.load(in_file)
                self.dataset.load('in', data_in)
        # Output
        self.current_out_partition_name = self.dataset_dir + self.list_out_partitions[self.mode][-1]
        with open(self.current_out_partition_name, 'rb') as out_file:
            out_size = os.fstat(out_file.fileno()).st_size
            while self.dataset.data_out.nbytes < out_size:
                out_size -= 128  # Each array takes 128 extra bytes in memory
                data_out = np.load(out_file)
                self.dataset.load('out', data_out)

    def getData(self, get_inputs, get_outputs, batch_size=1, batched=True, force_dataset_reload=False):
        if self.current_in_partition_name is None or self.dataset.currentSample >= len(self.dataset.data_in):
            if not force_dataset_reload:
                return None
            self.loadPartitions()
            if self.shuffle_dataset:
                self.dataset.shuffle()
            self.dataset.currentSample = 0
        idx = self.dataset.currentSample
        data = {'in': np.array([]), 'out': np.array([])}
        if get_inputs:
            if batched:
                data['in'] = self.dataset.data_in[idx: idx + batch_size].reshape((-1, *self.dataset.inShape))
            else:
                data['in'] = np.squeeze(self.dataset.data_in[idx: idx + batch_size], axis=0)
        if get_outputs:
            if batched:
                data['out'] = self.dataset.data_out[idx: idx + batch_size].reshape((-1, *self.dataset.outShape))
            else:
                data['out'] = np.squeeze(self.dataset.data_out[idx: idx + batch_size], axis=0)
        self.dataset.currentSample += batch_size
        return data

    def getNextBatch(self, batch_size):
        return self.getData(get_inputs=True, get_outputs=True, batch_size=batch_size, batched=True)

    def getNextSample(self, batched=True):
        return self.getData(get_inputs=True, get_outputs=True, batched=batched)

    def getNextInput(self, batched=False):
        return self.getData(get_inputs=True, get_outputs=False, batched=batched)

    def getNextOutput(self, batched=False):
        return self.getData(get_inputs=False, get_outputs=True, batched=batched)

    def loadPartitions(self):
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
        in_filenames, out_filenames = [], []
        for mode in modes:
            in_filenames += [self.dataset_dir + partition for partition in self.list_in_partitions[mode]]
            out_filenames += [self.dataset_dir + partition for partition in self.list_out_partitions[mode]]
        in_files = [open(filename, 'rb') for filename in in_filenames]
        out_files = [open(filename, 'rb') for filename in out_filenames]
        in_sizes = [os.stat(in_file.fileno()).st_size for in_file in in_files]
        out_sizes = [os.stat(out_file.fileno()).st_size for out_file in out_files]
        in_loaded = [0. for _ in range(len(in_sizes))]
        out_loaded = [0. for _ in range(len(out_sizes))]
        end_partition = False
        idx_file = 0
        while self.dataset.memory_size() < self.max_size and not end_partition:
            in_sizes[idx_file] -= 128
            data_in = np.load(in_files[idx_file])
            in_loaded[idx_file] += data_in.nbytes
            self.dataset.load('in', data_in)
            if in_loaded[idx_file] >= in_sizes[idx_file]:
                end_partition = True

            try:
                out_sizes[idx_file] -= 128
                data_out = np.load(out_files[idx_file])
                out_loaded[idx_file] += data_out.nbytes
                self.dataset.load('out', data_out)
                if out_loaded[idx_file] >= out_sizes[idx_file]:
                    end_partition = True
            except:
                pass

            idx_file = (idx_file + 1) % len(in_files)

    def close(self):
        if not self.saved:
            self.saveData()

    def getDescription(self):
        if len(self.description) == 0:
            self.description += "\nDATASET MANAGER:\n"
            self.description += "   Partition size: {}Go\n".format(self.max_size * 1e-9)
            self.description += "   Dataset path: {}\n".format(self.dataset_dir)
        return self.description
