import os
import shutil
import numpy as np
from sklearn.utils import shuffle

from DeepPhysX.Dataset.Dataset import Dataset
import DeepPhysX.utils.tensor_transform_utils as ttu


class DatasetManager:
    def __init__(self, dataset_dir, network_name, partition_size=1, read_only=False):
        # Path variables
        self.managerAbsolutePath = '{}/'.format(os.path.dirname(os.path.realpath(__file__)))
        self.datasetPath = dataset_dir
        self.datasetAbsolutePath = self.managerAbsolutePath + self.datasetPath
        # Network variables
        self.networkName = network_name
        # Partition variables
        self.partitionNameTemplate = self.datasetPath + self.networkName + '_{}_{}.npy'
        self.partitionsList = [[], [], []]
        self.partitionsListPath = [self.datasetPath + 'Dataset_train_partitions_list.txt',
                                   self.datasetPath + 'Dataset_test_partitions_list.txt']
        self.currentPartition = [0, 0]
        self.currentFile = None
        self.maxSize = int(partition_size * 1e9)  # from gigabytes to bytes
        self.currentSize = [0, 0]
        self.nbLoadedFiles = 0
        self.firstLoadedFile = ''
        self.in_and_out = True
        self.done_reading = False
        # Mode used to handle inputs and outputs : (on/off)line_(train/test)
        # Submode : train, test, run
        self.modesList = ['online_train', 'offline_train', 'online_predict', 'offline_predict']
        self.mode = None
        self.submode = None
        self.pureOffline = False
        # Data : big array made of all the input and output from a file
        self.data = {self.modesList[0]: [np.array([]), np.array([])],
                     self.modesList[1]: [np.array([]), np.array([])],
                     self.modesList[3]: [np.array([]), np.array([])],
                     self.modesList[4]: [np.array([]), np.array([])]}
        self.inputShape = None
        self.inputFlatShape = None
        self.outputShape = None
        self.outputFlatShape = None
        # Index
        self.indexBegin = 0
        self.indexStep = None
        # Init dataset
        self.initDataset(read_only)

    def initDataset(self, read_only):
        if not read_only:
            # Create the folder in which everything will be written
            if os.path.isdir(self.datasetAbsolutePath):
                shutil.rmtree(self.datasetAbsolutePath, ignore_errors=True)
            os.makedirs(self.datasetAbsolutePath)
            self.setTrainingMode()
            self.createNewPartition()
        else:
            # Look for files which ends with "partitions_list.txt"
            self.setOfflineTrainingMode()
            self.loadDirectory()
            self.setOfflineTrainingMode()

    def appendDataToFile(self, network_input, ground_truth):
        if self.mode != "online_train":
            return
        if (self.inputFlatShape is None) or (self.outputFlatShape) is None:
            self.inputShape = network_input[0].shape
            self.outputShape = ground_truth[0].shape
            self.inputFlatShape = len(ttu.flatten(network_input[0])) if len(network_input) > 0 else None
            self.outputFlatShape = len(ttu.flatten(ground_truth[0])) if len(ground_truth) > 0 else None
        np.save(self.currentFile[0], ttu.flatten(network_input))
        np.save(self.currentFile[1], ttu.flatten(ground_truth))
        path1 = self.managerAbsolutePath + self.partitionsList[self.submode][-1]
        path2 = self.managerAbsolutePath + self.partitionsList[self.submode][-2]
        if (os.path.getsize(path1) > self.maxSize) or (os.path.getsize(path2)) > self.maxSize:
            self.createNewPartition()

    def createNewPartition(self):
        current_file_path_in = self.partitionNameTemplate.format('IN', len(self.partitionsList[self.submode]) // 2)
        current_file_path_out = self.partitionNameTemplate.format('OUT', len(self.partitionsList[self.submode]) // 2)
        print('####')
        print('    A new partition has been created wih maximum size ~{}Gb'.format(float(self.maxSize) / 1e9))
        print('    Inputs: {}'.format(current_file_path_in))
        print('    Outputs: {}'.format(current_file_path_out))
        print('####')
        self.partitionsList[self.submode].append(current_file_path_in)
        self.partitionsList[self.submode].append(current_file_path_out)
        if self.currentFile is not None:
            if not self.currentFile[0].closed:
                self.currentFile[0].close()
            if not self.currentFile[1].closed:
                self.currentFile[1].close()
        self.currentFile = [open(self.managerAbsolutePath + current_file_path_in, 'ab'),
                            open(self.managerAbsolutePath + current_file_path_out, 'ab')]
        return

    def saveData(self):
        file = open(self.managerAbsolutePath + self.partitionsListPath[self.submode], "w+")
        for filepath in self.partitionsList[self.submode]:
            file.write(filepath + '\n')
        file.close()
        self.data[self.submode] = np.empty(0)

    def close(self):
        if (self.mode == self.modesList[0]) or (self.mode == self.modesList[2]):
            self.setTrainingMode()
            self.saveData()
            if len(self.partitionsList[1]) > 0:
                self.setTestMode()
                self.saveData()
        self.currentPartition = [0, 0]

    def offLineDataInitialisation(self, input_shape=None, output_shape=None):
        if (input_shape is None and self.input_shape is None) or (output_shape is None and self.output_shape is None):
            raise Exception("offLineDataInitialisation : For pure offline training, please provide input and output \
            shapes")
        if input_shape is not None:
            self.inputShape = input_shape
        if output_shape is not None:
            self.outputShape = output_shape
        self.inputFlatShape = np.prod(np.array(self.inputShape))
        self.outputFlatShape = np.prod(np.array(self.outputShape))
        self.indexStep = None
        self.close()
        self.setOffLineTrainingMode()
        self.loadDirectory()


    def loadPartition(self, shuffle_dataset=True, multiple_partitions=False):
        # Reset data
        if len(self.data[self.mode][0].shape) == 1:
            self.data[self.mode][0] = np.array([], dtype=np.float32).reshape((0, *self.inputShape))
        if len(self.data[self.mode][1].shape) == 1:
            self.data[self.mode][1] = np.array([], dtype=np.float32).reshape((0, *self.outputShape))
        # Index
        self.indexBegin, index_end = self.getNextIndex(multiple_partitions)
        # If "multiple partition" load each file, then if "in&out" load 2 files, then load 1 file
        nb_files_to_load = len(self.partitionsList[self.submode]) if multiple_partitions else 2 if self.in_and_out else 1
        loaded_files = 0
        in_out = 0
        # Check if there is still some files to read
        while loaded_files < nb_files_to_load:
            # Remove '\n' at the end of the line
            path = self.partitionsList[self.submode][self.currentPartition[self.submode]][:-1]
            # Read and reshape the data
            # Since we have multiple times into the numpy file, we have to load the same amount of array
            # Otherwise it will load the first array only
            shape = self.inputShape if in_out == 0 else self.outputShape
            with open(path, 'rb') as f:
                fsz = os.fstat(f.fileno()).st_size
                file_data = np.load(f)
                # Each load fetch one or all tensors from the file
                while file_data.shape[0] < index_end and f.tell() < fsz:
                    file_data = np.concatenate((file_data, np.load(f, allow_pickle=True)))
            # Update current status
            self.currentPartition[self.submode] += 1
            loaded_files += 1
            # Since the last tensor we red is index_end we don't need an upper bound
            file_data = file_data.reshape((-1, *shape))[min(file_data.shape[0], self.indexBegin):]
            if file_data.shape[0] == 0:
                break
            self.data[self.mode][in_out] = np.concatenate((self.data[self.mode][in_out], file_data))
            # Swap between input and output index
            in_out = (in_out + 1) % 2 if self.in_and_out else 0
        if multiple_partitions:
            self.currentPartition[self.mode] = 0
        if shuffle_dataset:
            self.data[self.mode][0], self.data[self.mode][1] = shuffle(self.data[self.mode][0], self.data[self.mode][1])

    def checkAlreadyRead(self, path):
        if self.nbLoadedFiles == 0:
            self.firstLoadedFile = path
        elif path == self.firstLoadedFile and self.nbLoadedFiles > 0:
            self.done_reading = True
        else:
            self.nbLoadedFiles += 1

    def getData(self, inputs, outputs, shuffle_dataset=True, multiple_partitions=True, batch_size=1, batched=True):
        if (self.data[self.mode][0].shape[0] < batch_size and inputs) or (self.data[self.mode][1].shape[0] < batch_size and outputs):
            self.loadPartition(shuffle_dataset=shuffle_dataset, multiple_partitions=multiple_partitions)
        out = []
        if inputs:
            if batched:
                out.append(self.data[self.mode][0][-batch_size:])
            else:
                out.append(np.squeeze(self.data[self.mode][0][-batch_size:], axis=0))
            self.data[self.mode][0] = self.data[self.mode][0][:-batch_size]
        if outputs:
            if batched:
                out.append(self.data[self.mode][1][-batch_size:])
            else:
                out.append(np.squeeze(self.data[self.mode][1][-batch_size:], axis=0))
            self.data[self.mode][1] = self.data[self.mode][1][:-batch_size]
        return out

    def getNextBatch(self, batch_size, shuffle_dataset=True, multiple_partitions=False):
        return self.getData(inputs=True, outputs=True, shuffle_dataset=shuffle_dataset,
                            multiple_partitions=multiple_partitions, batch_size=batch_size, batched=True)

    def getNextSample(self, shuffle_dataset=True, multiple_partitions=False, batched=True):
        return self.getData(inputs=True, outputs=True, shuffle_dataset=shuffle_dataset,
                            multiple_partitions=multiple_partitions, batched=batched)

    def getNextInput(self, shuffle_dataset=True, multiple_partitions=False, batched=False):
        return self.getData(inputs=True, outputs=False, shuffle_dataset=shuffle_dataset,
                            multiple_partitions=multiple_partitions, batched=batched)

    def getNextOutput(self, shuffle_dataset=True, multiple_partitions=False, batched=True):
        return self.getData(inputs=False, outputs=True, shuffle_dataset=shuffle_dataset,
                            multiple_partitions=multiple_partitions, batched=batched)

    def purge(self):
        self.data[self.mode][0] = np.empty(0)
        self.data[self.mode][1] = np.empty(0)
        self.currentPartition[self.submode] = 0

    def addPartitionToList(self, partition_list):
        parts = []
        for i in range(0, len(partition_list), 2):
            in_file = partition_list[i]
            if i + 1 < len(partition_list):
                out_file = partition_list[i+1]
            if 'IN' in in_file and 'OUT' in out_file:
                self.in_and_out &= True
                if not self.in_and_out:
                    print("At least one file name in the partitions list does not contain IN or OUT.")
                    print("May cause misalignment of the outputs with the input. Leaving the loading.")
                    quit(0)
            else:
                self.in_and_out = False
            parts.append(self.managerAbsolutePath + in_file)
            parts.append(self.managerAbsolutePath + out_file)
        return parts

    def loadDirectory(self, dir_name=''):
        # Load a directory according to the distribution given by teh lists
        # Or the whole directory as training data if not provided
        self.currentPartition[self.submode] = 0
        if dir_name == '':
            dir_name = self.datasetAbsolutePath
        print("Loading directory: {}".format(dir_name))
        if not os.path.isdir(dir_name):
            raise Exception('Loading directory: The given input is not an existing directory\n{}'.format(dir_name))
        # Look for while which ends with "partitions_list.txt"
        partition_list_file = [f for f in os.listdir(dir_name) if
                               os.isfile(os.join(dir_name, f)) and f.endswith('partitions_list.txt')]
        # If there is no such files then proceed to load any dataset found as in/out
        if len(partition_list_file) == 0:
            print("Will consider any .npy file as input/output.")
            dataset_list = [f for f in os.listdir(dir_name) if
                            os.isfile(os.join(dir_name, f)) and f.enswith(".npy")]
            self.partitionsList[self.submode] = self.addPartitionToList(dataset_list)
        else:
            # If partitions_list.txt found then proceed to load the specific dataset as input/output
            for p_l in partition_list_file:
                partition_file = open(self.datasetAbsolutePath + p_l)
                partition_list = partition_file.readlines()
                self.partitionsList[self.submode] = self.addPartitionToList(partition_list)

    def setTrainingMode(self):
        self.mode = self.modesList[0]
        self.submode = 0

    def setTestMode(self):
        self.mode = self.modesList[2]
        self.submode = 1

    def setOfflineTrainingMode(self):
        self.mode = self.modesList[1]
        self.submode = 0

    def setOfflineTestMode(self):
        self.mode = self.modesList[3]
        self.submode = 1

    def description(self, minimal=False):
        desc = "\n\nDataset Manager: \n"
        desc += "Partition size: {}Go\n".format(self.maxSize * 1e-9)
        desc += "Dataset path: {}\na".format(self.datasetAbsolutePath)
        return desc


    def getNextIndex(self, multiple_partition):
        if multiple_partition:
            if self.indexStep is None:
                path_in = self.partitionsList[self.submode][0].replace("\n", "")
                data_in = np.float32(np.load(path_in, allow_pickle=True)).reshape((-1, *self.input_shape))
                self.indexStep = int(data_in.shape[0] / len(self.partitionsList[self.submode]))
                return 0, self.indexStep
            return self.indexBegin + self.indexStep, self.indexBegin ++ 2 * self.index_step
        else:
            # Represent 1e100 tensor (huge value for the indexing)
            return 0, 1e100

