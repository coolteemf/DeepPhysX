import numpy as np


class Dataset:

    def __init__(self, max_size):
        # Data storage
        self.data = {'in': np.array([]), 'out': np.array([])}
        self.maxSize = max_size
        self.currentSample = 0
        # Sizes
        self.inShape = None
        self.inFlatShape = None
        self.outShape = None
        self.outFlatShape = None
        # Description
        self.description = ""

    def reset(self):
        self.data['in'] = np.array([]).reshape((0, self.inFlatShape)) if self.inFlatShape is not None else np.array([])
        self.data['out'] = np.array([]).reshape((0, self.outFlatShape)) if self.outFlatShape is not None else np.array([])
        self.currentSample = 0

    def getSize(self):
        return self.data['in'].nbytes, self.data['out'].nbytes

    def add(self, network_input, ground_truth):
        if (self.inFlatShape is None) or (self.outFlatShape is None):
            self.inShape = network_input[0].shape
            self.outShape = ground_truth[0].shape
            self.inFlatShape = len(network_input[0].flatten())
            self.outFlatShape = len(ground_truth[0].flatten())
        if len(self.data['in']) == 0:
            self.data['in'] = np.array([]).reshape((0, self.inFlatShape))
            self.data['out'] = np.array([]).reshape((0, self.outFlatShape))
        for i in range(len(network_input)):
            self.data['in'] = np.concatenate((self.data['in'], network_input[i].flatten()[None, :]))
            self.data['out'] = np.concatenate((self.data['out'], ground_truth[i].flatten()[None, :]))

    def loadData(self, network_input, ground_truth):
        if (self.inFlatShape is None) or (self.outFlatShape is None):
            self.inShape = network_input[0].shape
            self.outShape = ground_truth[0].shape
            self.inFlatShape = len(network_input[0].flatten())
            self.outFlatShape = len(ground_truth[0].flatten())
        if len(self.data['in']) == 0:
            self.data['in'] = np.array([]).reshape((0, self.inFlatShape,))
            self.data['out'] = np.array([]).reshape((0, self.outFlatShape))
        self.data['in'] = np.concatenate((self.data['in'], network_input), axis=0)
        self.data['out'] = np.concatenate((self.data['out'], ground_truth), axis=0)

    def shuffle(self):
        index = np.arange(len(self.data['in']))
        np.random.shuffle(index)
        network_input = np.copy(self.data['in'])
        ground_truth = np.copy(self.data['out'])
        for i in range(len(network_input)):
            network_input[i] = self.data['in'][index[i]]
            ground_truth[i] = self.data['out'][index[i]]
        self.data['in'] = network_input
        self.data['out'] = ground_truth

    def getDescription(self):
        if len(self.description) == 0:
            self.description += "\nCORE Dataset:\n"
            self.description += "   Max size: {}\n".format(self.maxSize)
            self.description += "   Input shape, input flat shape: {}, {}\n".format(self.inShape, self.inFlatShape)
            self.description += "   Output shape, output flat shape: {}, {}\n".format(self.outShape, self.outFlatShape)
        return self.description
