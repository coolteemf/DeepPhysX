from .Dataset import Dataset


class DatasetConfig:

    def __init__(self, partition_size, dataset_dir=None, generate_data=False, shuffle_dataset=False):

        self.dataset = Dataset
        self.maxSize = int(partition_size * 1e9)
        self.datasetConfig = self.maxSize

        self.datasetDir = dataset_dir
        self.existingDataset = False if dataset_dir is None else True
        self.generateData = generate_data

        self.shuffleDataset = shuffle_dataset

    def createDataset(self):
        return self.dataset(self.datasetConfig)
