import os
import numpy as np

from DeepPhysX.Dataset.BaseDatasetConfig import BaseDatasetConfig
import DeepPhysX.utils.pathUtils as pathUtils


class DatasetManager:

    def __init__(self, dataset_config: BaseDatasetConfig, session_name='default', session_dir=None, new_session=True,
                 train=True, record_data=None):

        self.name = self.__class__.__name__

        # Create the dataset
        self.dataset = dataset_config.createDataset()

        # Get dataset parameters
        self.max_size = self.dataset.max_size
        self.shuffle_dataset = dataset_config.shuffle_dataset
        self.record_data = record_data if record_data is not None else {'in': True, 'out': True}

        # Partition variables
        self.modes = {'training': 0, 'validation': 1, 'running': 2}
        self.mode = self.modes['training'] if train else self.modes['running']
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
        self.saved = False

        # Init dataset directories
        self.session_dir = session_dir if session_dir is not None else os.path.join(pathUtils.getFirstCaller(),
                                                                                    session_name)
        dataset_dir = dataset_config.dataset_dir

        # Training
        if train:
            if new_session:
                if dataset_dir is None:  # New dataset
                    self.dataset_dir = pathUtils.createDir(dirname=os.path.join(self.session_dir, 'dataset/'),
                                                           check_existing='dataset')
                    self.createNewPartitions()
                else:  # Train from another session's dataset
                    self.dataset_dir = pathUtils.copyDir(src_dir=dataset_dir,
                                                         dest_parent_dir=self.session_dir,
                                                         dest_dir='dataset')
                    self.loadDirectory()
            else:  # Train from this session's dataset
                self.dataset_dir = os.path.join(self.session_dir, 'dataset/')
                self.loadDirectory()
            # Need an environment if existing dataset with inputs without corresponding outputs
            self.create_environment = self.requireEnvironment()

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
            self.current_in_partition = open(self.current_in_partition_name, 'wb')

        if self.record_data['out']:
            current_part_out = self.partitions_templates[self.mode].format('OUT', self.idx_partitions[self.mode])
            print("               Outputs: {}".format(self.dataset_dir + current_part_out))
            self.list_out_partitions[self.mode].append(current_part_out)
            partitions_list_file.write(current_part_out + '\n')
            self.current_out_partition_name = self.dataset_dir + current_part_out
            self.current_out_partition = open(self.current_out_partition_name, 'wb')

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
        for mode in ['training', 'validation', 'running']:
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
            self.addPartitionsToList(partitions_list, mode)

        # Load the dataset
        self.loadPartitions()

    def addPartitionsToList(self, partitions_list, mode):
        nb_parts = len(partitions_list)
        partitions_in = sorted([f for f in partitions_list if f.__contains__('IN')])
        partitions_out = sorted([f for f in partitions_list if f.__contains__('OUT')])
        if nb_parts != len(partitions_in) + len(partitions_out):
            raise ValueError("[{}] The number of partitions is ambiguous.".format(self.name))
        self.list_in_partitions[self.modes[mode]] = partitions_in
        self.list_out_partitions[self.modes[mode]] = partitions_out

    def loadPartitions(self):
        if not self.generate_data:
            self.dataset.reset()
            # No partitions for the actual mode
            if len(self.partitions_lists[self.mode]) == 0:
                print("Load Partition: No partition found for {} mode".format(self.mode))
            # Single partition for the actual mode
            elif len(self.partitions_lists[self.mode]) == 1:
                self.loadLastPartitions()
            # Multiple partitions
            else:
                for i in range(len(self.partitions_lists[self.mode]) // 2):
                    self.current_partition_names = {'in': self.dataset_dir + self.partitions_lists[self.mode][2 * i],
                                                    'out': self.dataset_dir + self.partitions_lists[self.mode][
                                                        2 * i + 1]}
                    data_in = np.load(self.current_partition_names['in'])
                    data_out = np.load(self.current_partition_names['out'])
                    self.dataset.load(data_in, data_out)
            if self.shuffle_dataset:
                self.dataset.shuffle()

    def loadLastPartitions(self):

        if self.record_data['in']:
            self.current_in_partition_name = self.dataset_dir + self.list_in_partitions[self.mode][-1]
            data_in = np.load(self.current_partition_names['in'])
            self.dataset.load('in', data_in)

        if self.record_data['out']:
            self.current_out_partition_name = self.dataset_dir + self.list_out_partitions[self.mode][-1]
            data_out = np.load(self.current_partition_names['out'])
            self.dataset.load('out', data_out)

    def requireEnvironment(self):
        # Called while training to check if each inputs as an output, otherwise need an environment to compute it
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
        if self.record_data['in'] or self.record_data['out']:
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
        else:
            self.mode = mode
            self.dataset.reset()
            self.loadPartitions()

    def close(self):
        if not self.saved:
            self.saveData()

    def getData(self, get_inputs, get_outputs, batch_size=1, batched=True):
        if self.dataset.currentSample >= len(self.dataset.data['in']):
            self.dataset.shuffle()
            self.dataset.currentSample = 0
        idx = self.dataset.currentSample
        data = {'in': np.array([]), 'out': np.array([])}
        if get_inputs:
            if batched:
                data['in'] = self.dataset.data['in'][idx: idx + batch_size]
            else:
                data['in'] = np.squeeze(self.dataset.data['in'][idx: idx + batch_size], axis=0)
        if get_outputs:
            if batched:
                data['out'] = self.dataset.data['out'][idx: idx + batch_size]
            else:
                data['out'] = np.squeeze(self.dataset.data['out'][idx: idx + batch_size], axis=0)
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

    def getDescription(self):
        if len(self.description) == 0:
            self.description += "\nDATASET MANAGER:\n"
            self.description += "   Partition size: {}Go\n".format(self.max_size * 1e-9)
            self.description += "   Dataset path: {}\n".format(self.dataset_dir)
        return self.description
