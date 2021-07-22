import os
from dataclasses import dataclass

from DeepPhysX_Core.Dataset.BaseDataset import BaseDataset


class BaseDatasetConfig:

    @dataclass
    class BaseDatasetProperties:
        max_size: int

    def __init__(self, dataset_class=BaseDataset, dataset_dir=None, partition_size=1., shuffle_dataset=False):
        """
        BaseDatasetConfig is a configuration class to parameterize and create a BaseDataset for the DatasetManager.

        :param BaseDataset dataset_class: BaseDataset class from which an instance will be created
        :param str dataset_dir: Name of an existing dataset repository
        :param float partition_size: Maximum size in Gb of a single dataset partition
        :param bool shuffle_dataset: Specify if existing dataset should be shuffled
        """

        self.name = self.__class__.__name__

        # Check dataset_dir type and existence
        if dataset_dir is not None:
            if type(dataset_dir) != str:
                raise TypeError(f"[{self.name}] Wrong dataset_dir type: str required, get {type(dataset_dir)}")
            if not os.path.isdir(dataset_dir):
                raise ValueError(f"[{self.name}] Given dataset_dir doesn't exists: {dataset_dir}")
        # Check partition_size type and value
        if type(partition_size) != int and type(partition_size) != float:
            raise TypeError(f"[{self.name}] Wrong partition_size type: float required, get {type(partition_size)}")
        if partition_size <= 0:
            raise ValueError(f"[{self.name}] Given partition_size is negative or null")
        # Check shuffle_dataset type
        if type(shuffle_dataset) != bool:
            raise TypeError(f"[{self.name}] Wrong shuffle_dataset type: bool required, get {type(shuffle_dataset)}")

        # BaseDataset parameterization
        self.dataset_class = dataset_class
        self.dataset_config = self.BaseDatasetProperties(max_size=int(partition_size * 1e9))

        # DatasetManager parameterization
        self.dataset_dir = dataset_dir
        self.shuffle_dataset = shuffle_dataset

    def createDataset(self):
        """
        :return: BaseDataset object from dataset_class and its parameters
        """
        try:
            dataset = self.dataset_class(config=self.dataset_config)
        except:
            raise ValueError(f"[{self.name}] Given dataset_class got an unexpected keyword argument 'config'")
        if not isinstance(dataset, BaseDataset):
            raise TypeError(f"[{self.name}] Wrong dataset_class type: BaseDataset required, get {self.dataset_class}")
        return dataset

    def __str__(self):
        """
        :return: String containing information about the BaseDatasetConfig object
        """
        description = "\n"
        description += f"{self.name}\n"
        description += f"    Dataset class: {self.dataset_class.__name__}\n"
        description += f"    Max size: {self.dataset_config.max_size}\n"
        description += f"    Dataset dir: {self.dataset_dir}\n"
        description += f"    Shuffle dataset: {self.shuffle_dataset}\n"
        return description
