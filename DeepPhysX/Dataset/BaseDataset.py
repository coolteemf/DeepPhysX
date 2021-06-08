import numpy as np


class BaseDataset:

    def __init__(self, config):
        # Data storage
        self.data_in, self.data_out = np.array([]), np.array([])
        self.maxSize = config.max_size
        self.currentSample = 0
        # Sizes
        self.inShape, self.inFlatShape = None, None
        self.outShape, self.outFlatShape = None, None
        # Description
        self.descriptionName = "BaseDataset"
        self.description = ""

    def init_data_size(self, data_input, data_output):
        self.inShape, self.outShape = data_input.shape, data_output.shape
        self.inFlatShape, self.outFlatShape = len(data_input.flatten()), len(data_output.flatten())
        self.data_in = np.array([]).reshape((0, self.inFlatShape))
        self.data_out = np.array([]).reshape((0, self.outFlatShape))

    def reset(self):
        self.data_in = np.array([]).reshape((0, self.inFlatShape)) if self.inFlatShape is not None else np.array([])
        self.data_out = np.array([]).reshape((0, self.outFlatShape)) if self.outFlatShape is not None else np.array([])
        self.currentSample = 0

    def memory_size(self):
        return self.data_in.nbytes + self.data_out.nbytes

    def check_data(self, data_inputs, data_outputs):
        if type(data_inputs) != np.ndarray:
            raise TypeError("[BASEDATASET] Loaded inputs must be numpy array, {} found.".format(type(data_inputs)))
        if type(data_outputs) != np.ndarray:
            raise TypeError("[BASEDATASET] Loaded outputs must be numpy array, {} found.".format(type(data_outputs)))
        if len(data_inputs.shape) < 2:
            raise ValueError("[BASEDATASET] Loaded inputs shape must be dim > 2, {} found.".format(type(data_inputs)))
        if len(data_outputs.shape) < 2:
            raise ValueError("[BASEDATASET] Loaded outputs shape must be dim > 2, {} found.".format(type(data_outputs)))

    def add(self, data_inputs, data_outputs, partition_files):
        self.check_data(data_inputs, data_outputs)
        if (self.inFlatShape is None) or (self.outFlatShape is None):
            self.init_data_size(data_inputs[0], data_outputs[0])
        for i in range(len(data_inputs)):
            self.data_in = np.concatenate((self.data_in, data_inputs[i].flatten()[None, :]))
            np.save(partition_files['in'], data_inputs[i].flatten())
            self.data_out = np.concatenate((self.data_out, data_outputs[i].flatten()[None, :]))
            np.save(partition_files['out'], data_outputs[i].flatten())

    def load(self, data_inputs, data_outputs):
        self.check_data(data_inputs, data_outputs)
        if (self.inFlatShape is None) or (self.outFlatShape is None):
            self.init_data_size(data_inputs[0], data_outputs[0])
        self.data_in = np.concatenate((self.data_in, data_inputs), axis=0)
        self.data_out = np.concatenate((self.data_out, data_outputs), axis=0)

    def shuffle(self):
        index = np.arange(len(self.data_in))
        np.random.shuffle(index)
        inputs = np.copy(self.data_in)
        outputs = np.copy(self.data_out)
        for i in range(len(inputs)):
            inputs[i] = self.data_in[index[i]]
            outputs[i] = self.data_out[index[i]]
        self.data_in = inputs
        self.data_out = outputs

    def getDescription(self):
        if len(self.description) == 0:
            self.description += "\n{}\n".format(self.descriptionName)
            self.description += "   Max size: {}\n".format(self.maxSize)
            self.description += "   Input shape, input flat shape: {}, {}\n".format(self.inShape, self.inFlatShape)
            self.description += "   Output shape, output flat shape: {}, {}\n".format(self.outShape, self.outFlatShape)
        return self.description
