import os
import inspect
import shutil
import numpy as np

from DeepPhysX.Dataset.Dataset import Dataset


class DatasetManager:

    def __init__(self, network_name, partition_size=1, mode='train', shuffle_dataset=False, generate_data=True):
        # Dataset variables
        frm = inspect.stack()[1]
        mod = inspect.getmodule(frm[0])
        caller_path = os.path.dirname(os.path.abspath(mod.__file__))
        self.datasetPath = os.path.join(caller_path, network_name, 'dataset/')
        self.maxSize = int(partition_size * 1e9)  # from gigabytes to bytes
        self.dataset = Dataset(max_size=self.maxSize)
        self.generateData = generate_data
        self.mode = mode
        self.shuffleDataset = shuffle_dataset
        # Partition variables
        self.partitionsTemplates = {'train': network_name + '_train_{}_{}.npy',
                                    'test': network_name + '_test_{}_{}.npy',
                                    'predict': network_name + '_predict_{}_{}.npy'}
        self.partitionsLists = {'train': [], 'test': [], 'predict': []}
        self.partitionsListsFiles = {'train': self.datasetPath + 'Dataset_train_partitions_list.txt',
                                     'test': self.datasetPath + 'Dataset_test_partitions_list.txt',
                                     'predict': self.datasetPath + 'Dataset_predict_partitions_list.txt'}
        self.actualPartitions = {'train': 0, 'test': 0, 'predict': 0}
        self.currentPartitions = None
        self.saved = True
        # Uncertain variables
        self.in_and_out = True
        self.done_reading = False
        # Index
        self.indexBegin = 0
        self.indexStep = None
        # Init dataset
        self.initDataset()

    def initDataset(self):
        if self.generateData:
            # Create the folder in which everything will be written
            if os.path.isdir(self.datasetPath):
                shutil.rmtree(self.datasetPath, ignore_errors=True)
            os.makedirs(self.datasetPath)
            self.createNewPartitions()
        else:
            # Look for files which ends with "partitions_list.txt"
            self.loadDirectory()

    def createNewPartitions(self):
        # Create in and out partitions
        current_part_in = self.partitionsTemplates[self.mode].format('IN', self.actualPartitions[self.mode])
        current_part_out = self.partitionsTemplates[self.mode].format('OUT', self.actualPartitions[self.mode])
        self.actualPartitions[self.mode] += 1
        print("New Partition: A new partition has been created wih max size ~{}Gb".format(float(self.maxSize) / 1e9))
        print("               Inputs: {}".format(self.datasetPath + current_part_in))
        print("               Outputs: {}".format(self.datasetPath + current_part_out))
        # Add partitions to list
        self.partitionsLists[self.mode].append(current_part_in)
        self.partitionsLists[self.mode].append(current_part_out)
        partitions_list_file = open(self.partitionsListsFiles[self.mode], 'a')
        partitions_list_file.write(current_part_in + '\n' + current_part_out + '\n')
        partitions_list_file.close()
        # Keep current partitions
        self.currentPartitions = {'in': self.datasetPath + current_part_in,
                                  'out': self.datasetPath + current_part_out}

    def loadDirectory(self):
        # Load a directory according to the distribution given by the lists
        print("Loading directory: Read dataset from {}".format(self.datasetPath))
        if not os.path.isdir(self.datasetPath):
            raise Warning("Loading directory: The given path is not an existing directory")
        # Look for file which ends with 'partitions_list.txt'
        for mode in ['train', 'test', 'predict']:
            partitions_list_file = [f for f in os.listdir(self.datasetPath) if
                                    os.path.isfile(os.path.join(self.datasetPath, f)) and
                                    f.endswith(mode + '_partitions_list.txt')]
            # If there is no such files then proceed to load any dataset found as in/out
            if len(partitions_list_file) == 0:
                print("Loading directory: Partitions list not found for {} mode, will consider any .npy file as "
                      "input/output.".format(mode))
                partitions_list = [f for f in os.listdir(self.datasetPath) if
                                   os.path.isfile(os.path.join(self.datasetPath, f)) and f.endswith(".npy") and
                                   f.__contains__(mode)]
                self.partitionsLists[mode] = self.addPartitionsToList(partitions_list, from_file=False)
            # If partitions_list.txt found then proceed to load the specific dataset as input/output
            else:
                reader = open(self.datasetPath + partitions_list_file[0])
                partitions_list = reader.read().splitlines()
                reader.close()
                self.partitionsLists[mode] = self.addPartitionsToList(partitions_list, from_file=True)
        # Load the dataset
        self.loadPartitions()

    def addPartitionsToList(self, partitions_list, from_file):
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

    def addData(self, network_input, ground_truth):
        if self.generateData:
            # Check whether if there is still available place in partitions
            actual_size = self.dataset.getSize()
            if actual_size[0] > self.maxSize or actual_size[1] > self.maxSize:
                print("Saving through 'addData'")
                self.saveData()
                self.createNewPartitions()
                self.dataset.reset()
            self.dataset.add(network_input, ground_truth)
            self.saved = False

    def saveData(self):
        in_partition_file = open(self.currentPartitions['in'], 'wb')
        out_partition_file = open(self.currentPartitions['out'], 'wb')
        np.save(in_partition_file, self.dataset.data['in'])
        np.save(out_partition_file, self.dataset.data['out'])
        self.saved = True
        in_partition_file.close()
        out_partition_file.close()

    def setMode(self, mode):
        # Nothing has to be done if you do not change mode
        if mode == self.mode:
            return
        if self.generateData:
            # Save dataset before changing mode
            if not self.saved:
                self.saveData()
            self.mode = mode
            self.dataset.reset()
            # Create or load partition for the new mode
            if self.actualPartitions[self.mode] == 0:
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
        if self.generateData and not self.saved:
            print("Saving through 'close'")
            self.saveData()

    def loadLastPartitions(self):
        self.currentPartitions = {'in': self.datasetPath + self.partitionsLists[self.mode][-2],
                                  'out': self.datasetPath + self.partitionsLists[self.mode][-1]}
        data_in = np.load(self.currentPartitions['in'])
        data_out = np.load(self.currentPartitions['out'])
        self.dataset.loadData(data_in, data_out)

    def loadPartitions(self):
        if not self.generateData:
            self.dataset.reset()
            # No partitions for the actual mode
            if len(self.partitionsLists[self.mode]) == 0:
                print("Load Partition: No partition found for {} mode".format(self.mode))
            # Single partition for the actual mode
            elif len(self.partitionsLists[self.mode]) == 1:
                self.loadLastPartitions()
            # Multiple partitions
            else:
                for i in range(len(self.partitionsLists[self.mode]) // 2):
                    self.currentPartitions = {'in': self.datasetPath + self.partitionsLists[self.mode][2 * i],
                                              'out': self.datasetPath + self.partitionsLists[self.mode][2 * i + 1]}
                    data_in = np.load(self.currentPartitions['in'])
                    data_out = np.load(self.currentPartitions['out'])
                    self.dataset.loadData(data_in, data_out)
            if self.shuffleDataset:
                self.dataset.shuffle()

    def getData(self, inputs, outputs, batch_size=1, batched=True):
        if self.dataset.currentSample > len(self.dataset.data['in']):
            self.dataset.shuffle()
            self.dataset.currentSample = 0
        idx = self.dataset.currentSample
        res = {'in': np.array([]), 'out': np.array([])}
        if inputs:
            if batched:
                res['in'] = self.dataset.data['in'][idx: idx + batch_size]
            else:
                res['in'] = np.squeeze(self.dataset.data['in'][idx: idx + batch_size], axis=0)
        if outputs:
            if batched:
                res['out'] = self.dataset.data['out'][idx: idx + batch_size]
            else:
                res['out'] = np.squeeze(self.dataset.data['out'][idx: idx + batch_size], axis=0)
        self.dataset.currentSample += 1
        return res

    def getNextBatch(self, batch_size):
        return self.getData(inputs=True, outputs=True, batch_size=batch_size, batched=True)

    def getNextSample(self, batched=True):
        return self.getData(inputs=True, outputs=True, batched=batched)

    def getNextInput(self, batched=False):
        return self.getData(inputs=True, outputs=False, batched=batched)

    def getNextOutput(self, batched=False):
        return self.getData(inputs=False, outputs=True, batched=batched)

    def description(self):
        desc = "\n\nDataset Manager: \n"
        desc += "Partition size: {}Go\n".format(self.maxSize * 1e-9)
        desc += "Dataset path: {}\na".format(self.datasetPath)
        return desc
