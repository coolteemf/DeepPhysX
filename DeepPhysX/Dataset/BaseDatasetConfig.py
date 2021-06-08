from dataclasses import dataclass

from DeepPhysX.Dataset.BaseDataset import BaseDataset


class BaseDatasetConfig:

    @dataclass
    class BaseDatasetProperties:
        max_size: float

    def __init__(self, dataset_class=BaseDataset, dataset_dir=None, partition_size=1., generate_data=False,
                 shuffle_dataset=False):

        # Check the arguments before to configure anything
        if dataset_dir is not None and type(dataset_dir) != str:
            raise TypeError("[BASEDATASETCONFIG] The dataset directory must be an str.")
        if type(partition_size) != int and type(partition_size) != float:
            raise TypeError("[BASEDATASETCONFIG] The partition size must be an int or a float.")
        if type(generate_data) != bool:
            raise TypeError("[BASEDATASETCONFIG] The generate data variable must be an boolean.")
        if type(shuffle_dataset) != bool:
            raise TypeError("[BASEDATASETCONFIG] The shuffle data variable must be an boolean.")

        # Dataset class
        self.dataset_class = dataset_class
        # Dataset configuration
        self.__dataset_config = self.BaseDatasetProperties(max_size=int(partition_size * 1e9))
        # DatasetManager configuration
        self.dataset_dir = dataset_dir
        self.existing_dataset = False if dataset_dir is None else True
        self.generate_data = generate_data
        self.shuffle_dataset = shuffle_dataset

        # Description
        self.description_name = "BaseDatasetConfig"
        self.description = ""

    def createDataset(self):
        try:
            dataset = self.dataset_class(self.__dataset_config)
        except:
            raise TypeError("[BASEDATASETCONFIG] The given dataset class is not a BaseDataset child class.")
        if not isinstance(dataset, BaseDataset):
            raise TypeError("[BASEDATASETCONFIG] The dataset class must be a BaseDataset child object.")
        return dataset

    def getDescription(self):
        if len(self.description) == 0:
            self.description += "\n{}\n".format(self.description_name)
            self.description += "   (dataset) Dataset class: {}\n".format(self.dataset_class.__name__)
            self.description += "   (dataset) Max size: {}\n".format(self.__dataset_config.max_size)
            self.description += "   (dataset) Dataset config: {}\n".format(self.__dataset_config)
            self.description += "   (datasetManager) Dataset dir: {}\n".format(self.dataset_dir)
            self.description += "   (datasetManager) Existing dataset: {}\n".format(self.existing_dataset)
            self.description += "   (datasetManager) Generate data: {}\n".format(self.generate_data)
            self.description += "   (datasetManager) Shuffle dataset: {}\n".format(self.shuffle_dataset)
        return self.description
