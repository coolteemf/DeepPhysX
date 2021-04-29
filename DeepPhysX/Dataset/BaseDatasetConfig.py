from .BaseDataset import BaseDataset
from dataclasses import dataclass


class BaseDatasetConfig:

    @dataclass
    class BaseDatasetProperties:
        max_size: int

    def __init__(self, dataset_class=BaseDataset, partition_size=1, dataset_dir=None, generate_data=False,
                 shuffle_dataset=False):
        # Dataset configuration
        max_size = int(partition_size * 1e9)
        self.datasetConfig = self.BaseDatasetProperties(max_size=max_size)
        # Dataset variables
        self.dataset_class = dataset_class
        # DatasetManager variables
        self.datasetDir = dataset_dir
        self.existingDataset = False if dataset_dir is None else True
        self.generateData = generate_data
        self.shuffleDataset = shuffle_dataset
        # Description
        self.descriptionName = "CORE BaseDatasetConfig"
        self.description = ""

    def setDatasetProperties(self):
        return

    def createDataset(self):
        return self.dataset_class(self.datasetConfig)

    def getDescription(self):
        if len(self.description) == 0:
            self.description += "\n{}\n".format(self.descriptionName)
            self.description += "   (dataset) Dataset class: {}\n".format(self.dataset_class.__name__)
            self.description += "   (dataset) Max size: {}\n".format(self.datasetConfig.max_size)
            self.description += "   (dataset) Dataset config: {}\n".format(self.datasetConfig)
            self.description += "   (datasetManager) Dataset dir: {}\n".format(self.datasetDir)
            self.description += "   (datasetManager) Existing dataset: {}\n".format(self.existingDataset)
            self.description += "   (datasetManager) Generate data: {}\n".format(self.generateData)
            self.description += "   (datasetManager) Shuffle dataset: {}\n".format(self.shuffleDataset)
        return self.description
