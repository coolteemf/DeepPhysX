from dataclasses import dataclass

from DeepPhysX_Core.Dataset.BaseDataset import BaseDataset


class BaseDatasetConfig:

    @dataclass
    class BaseDatasetProperties:
        max_size: float

    def __init__(self, dataset_class=BaseDataset, dataset_dir=None, partition_size=1., shuffle_dataset=False):

        # Description
        self.name = self.__class__.__name__
        self.description = ""

        # Check the arguments before to configure anything
        if dataset_dir is not None and type(dataset_dir) != str:
            raise TypeError("[{}] The dataset directory must be an str.".format(self.name))
        if type(partition_size) != int and type(partition_size) != float:
            raise TypeError("[{}] The partition size must be an int or a float.".format(self.name))
        if type(shuffle_dataset) != bool:
            raise TypeError("[{}] The shuffle data variable must be an boolean.".format(self.name))

        # Dataset class
        self.dataset_class = dataset_class
        # Dataset configuration
        self.__dataset_config = self.BaseDatasetProperties(max_size=int(partition_size * 1e9))
        # DatasetManager configuration
        self.dataset_dir = dataset_dir
        self.shuffle_dataset = shuffle_dataset

    def createDataset(self):
        try:
            dataset = self.dataset_class(self.__dataset_config)
        except:
            raise TypeError("[{}] The given dataset class is not a BaseDataset child class.".format(self.name))
        if not isinstance(dataset, BaseDataset):
            raise TypeError("[{}] The dataset class must be a BaseDataset child object.".format(self.name))
        return dataset

    def getDescription(self):
        if len(self.description) == 0:
            self.description += "\n{}\n".format(self.name)
            self.description += "   (dataset) Dataset class: {}\n".format(self.dataset_class.__name__)
            self.description += "   (dataset) Max size: {}\n".format(self.__dataset_config.max_size)
            self.description += "   (dataset) Dataset config: {}\n".format(self.__dataset_config)
            self.description += "   (datasetManager) Dataset dir: {}\n".format(self.dataset_dir)
            self.description += "   (datasetManager) Shuffle dataset: {}\n".format(self.shuffle_dataset)
        return self.description
