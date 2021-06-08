import os
import numpy as np

from DeepPhysX.Dataset.BaseDatasetConfig import BaseDatasetConfig
import DeepPhysX.utils.pathUtils as pathUtils


class DatasetManager:

    def __init__(self, dataset_config=BaseDatasetConfig(), session_name='default', session_dir=None, train=True):

        # Create the dataset
        self.dataset = dataset_config.createDataset()

        # Get dataset parameters
        self.max_size = self.dataset.max_size
        self.shuffle_dataset = dataset_config.shuffle_dataset
        self.generate_data = dataset_config.generate_data

        # Manage repositories
        self.session_dir = session_dir if session_dir is not None else os.path.join(pathUtils.getFirstCaller(),
                                                                                    session_name)
        if dataset_config.existing_dataset:
            self.dataset_dir = pathUtils.copyDir(src_dir=dataset_config.dataset_dir, dest_parent_dir=self.session_dir,
                                                 dest_dir='dataset')
        else:
            self.dataset_dir = pathUtils.createDir(dirname=os.path.join(self.session_dir, 'dataset/'),
                                                   check_existing='dataset')

        # Partition variables
        self.modes = {'training': 0, 'validation': 1, 'experience': 2}
        self.mode = self.modes['training'] if train else self.modes['experience']
        self.partitions_templates = (session_name + '_training_{}_{}.npy',
                                     session_name + '_validation_{}_{}.npy',
                                     session_name + '_experience_{}_{}.npy')
        self.partitions_list_files = (self.dataset_dir + 'Training_partitions.txt',
                                      self.dataset_dir + 'Validation_partitions.txt',
                                      self.dataset_dir + 'Experience_partitions.txt')
        self.partitions_lists = [[], [], []]
        self.actual_partitions = [0, 0, 0]
        self.current_partitions = None
        self.current_files = None
        self.saved = False
        self.in_and_out = True

        # Init dataset
        if dataset_config.existing_dataset:
            self.loadDirectory()
        else:
            self.createNewPartitions()

        # Description
        self.description = ""

    def createNewPartitions(self):
        # Create in and out partitions
        current_part_in = self.partitions_templates[self.mode].format('IN', self.actual_partitions[self.mode])
        current_part_out = self.partitions_templates[self.mode].format('OUT', self.actual_partitions[self.mode])
        self.actual_partitions[self.mode] += 1
        print("New Partition: A new partition has been created with max size ~{}Gb".format(float(self.max_size) / 1e9))
        print("               Inputs: {}".format(self.dataset_dir + current_part_in))
        print("               Outputs: {}".format(self.dataset_dir + current_part_out))
        # Add partitions to list
        self.partitions_lists[self.mode].append(current_part_in)
        self.partitions_lists[self.mode].append(current_part_out)
        partitions_list_file = open(self.partitions_list_files[self.mode], 'a')
        partitions_list_file.write(current_part_in + '\n' + current_part_out + '\n')
        partitions_list_file.close()
        # Keep current partitions
        self.current_partitions = {'in': self.dataset_dir + current_part_in,
                                   'out': self.dataset_dir + current_part_out}
        self.current_files = {'in': open(self.current_partitions['in'], 'wb'),
                              'out': open(self.current_partitions['out'], 'wb')}

    def loadDirectory(self):
        # Load a directory according to the distribution given by the lists
        print("Loading directory: Read dataset from {}".format(self.dataset_dir))
        if not os.path.isdir(self.dataset_dir):
            raise Warning("Loading directory: The given path is not an existing directory")
        # Look for file which ends with 'partitions.txt'
        for mode in ['training', 'validation', 'experience']:
            partitions_list_file = [f for f in os.listdir(self.dataset_dir) if
                                    os.path.isfile(os.path.join(self.dataset_dir, f)) and
                                    f.endswith(mode + '_partitions.txt')]
            # If there is no such files then proceed to load any dataset found as in/out
            if len(partitions_list_file) == 0:
                print("Loading directory: Partitions list not found for {} mode, will consider any .npy file as "
                      "input/output.".format(mode))
                partitions_list = [f for f in os.listdir(self.dataset_dir) if
                                   os.path.isfile(os.path.join(self.dataset_dir, f)) and f.endswith(".npy") and
                                   f.__contains__(mode)]
                self.partitions_lists[self.modes[mode]] = self.addPartitionsToList(partitions_list, from_file=False)
            # If partitions_list.txt found then proceed to load the specific dataset as input/output
            else:
                reader = open(self.dataset_dir + partitions_list_file[0])
                partitions_list = reader.read().splitlines()
                reader.close()
                self.partitions_lists[self.modes[mode]] = self.addPartitionsToList(partitions_list)
        # Load the dataset
        self.loadPartitions()

    def addPartitionsToList(self, partitions_list, from_file=True):
        nb_parts = len(partitions_list)
        if nb_parts % 2 != 0:
            raise Warning("Add Partitions: there must be a pair number of partitions")
        # os.listdir has an arbitrary order, sort in the order we want
        if not from_file:
            partitions_list_sort = sorted(partitions_list)
            for i in range(nb_parts // 2):
                partitions_list[2 * i] = partitions_list_sort[i]
                partitions_list[2 * i + 1] = partitions_list_sort[i + nb_parts // 2]
        print(partitions_list)
        # Assert there is a IN and OUT file for each pair of partitions
        for i in range(0, nb_parts, 2):
            in_file = partitions_list[i]
            out_file = partitions_list[i + 1]
            if 'IN' in in_file and 'OUT' in out_file:
                self.in_and_out &= True
                if not self.in_and_out:
                    raise Warning("Add Partitions: At least one file in the partitions list does not contain IN/OUT.")
            else:
                self.in_and_out = False
        return partitions_list

    def addData(self, data):
        if self.generate_data:
            # Check whether if there is still available place in partitions
            if self.dataset.memory_size() > self.max_size:
                self.saveData()
                self.createNewPartitions()
                self.dataset.reset()
            network_input = data['in']
            ground_truth = data['out']
            self.dataset.add(network_input, ground_truth, self.current_files)
            self.saved = False

    def saveData(self):
        self.saved = True
        self.current_files['in'].close()
        self.current_files['out'].close()

    def setMode(self, mode):
        # Nothing has to be done if you do not change mode
        if mode == self.mode:
            return
        if self.generate_data:
            # Save dataset before changing mode
            if not self.saved:
                self.saveData()
            self.mode = mode
            self.dataset.reset()
            # Create or load partition for the new mode
            if self.actual_partitions[self.mode] == 0:
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
        if self.generate_data and not self.saved:
            self.saveData()

    def loadLastPartitions(self):
        self.current_partitions = {'in': self.dataset_dir + self.partitions_lists[self.mode][-2],
                                   'out': self.dataset_dir + self.partitions_lists[self.mode][-1]}
        data_in = np.load(self.current_partitions['in'])
        data_out = np.load(self.current_partitions['out'])
        self.dataset.load(data_in, data_out)

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
                    self.current_partitions = {'in': self.dataset_dir + self.partitions_lists[self.mode][2 * i],
                                               'out': self.dataset_dir + self.partitions_lists[self.mode][2 * i + 1]}
                    data_in = np.load(self.current_partitions['in'])
                    data_out = np.load(self.current_partitions['out'])
                    self.dataset.load(data_in, data_out)
            if self.shuffle_dataset:
                self.dataset.shuffle()

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
