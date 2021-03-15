import numpy as np


class Dataset:

    def __init__(self, max_size):
        # TODO : set private attributes
        self.data = {'in': np.array([]), 'out': np.array([])}
        self.inShape = None
        self.inFlatShape = None
        self.outShape = None
        self.outFlatShape = None
        self.maxSize = max_size
        self.currentSample = 0

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

    def loadData(self, data_in, data_out):
        if (self.inFlatShape is None) or (self.outFlatShape is None):
            self.inShape = data_in[0].shape
            self.outShape = data_out[0].shape
            self.inFlatShape = len(data_in[0].flatten())
            self.outFlatShape = len(data_out[0].flatten())
        if len(self.data['in']) == 0:
            self.data['in'] = np.array([]).reshape((0, self.inFlatShape))
            self.data['out'] = np.array([]).reshape((0, self.outFlatShape))
        self.data['in'] = np.concatenate((self.data['in'], data_in))
        self.data['out'] = np.concatenate((self.data['out'], data_out))

    def shuffle(self):
        index = np.arange(len(self.data['in']))
        np.random.shuffle(index)
        data_in = np.copy(self.data['in'])
        data_out = np.copy(self.data['out'])
        for i in range(len(data_in)):
            data_in[i] = self.data['in'][index[i]]
            data_out[i] = self.data['out'][index[i]]
        self.data['in'] = data_in
        self.data['out'] = data_out
