from os.path import isdir
from collections import namedtuple
from typing import Any

from DeepPhysX_Core.Dataset.BaseDataset import BaseDataset


class BaseDatasetConfig:

    def __init__(self,
                 dataset_class=BaseDataset,
                 dataset_dir: str = None,
                 partition_size: float = 1.,
                 shuffle_dataset: bool = True):
        """
        BaseDatasetConfig is a configuration class to parameterize and create a BaseDataset for the DatasetManager.

        :param dataset_class: BaseDataset class from which an instance will be created
        :type dataset_class: BaseDataset
        :param str dataset_dir: Name of an existing dataset repository
        :param float partition_size: Maximum size in Gb of a single dataset partition
        :param bool shuffle_dataset: Specify if existing dataset should be shuffled
        """

        self.name: str = self.__class__.__name__

        # Check dataset_dir type and existence
        if dataset_dir is not None:
            if type(dataset_dir) != str:
                raise TypeError(f"[{self.name}] Wrong dataset_dir type: str required, get {type(dataset_dir)}")
            if not isdir(dataset_dir):
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
        self.dataset_class: BaseDataset = dataset_class
        self.dataset_config: namedtuple = self.make_config(max_size=int(partition_size * 1e9))

        # DatasetManager parameterization
        self.dataset_dir: str = dataset_dir
        self.shuffle_dataset: bool = shuffle_dataset

    def make_config(self, **kwargs):
        """
        Create a namedtuple which gathers all the parameters for the Dataset configuration.
        For a child config class, only new items are required since parent's items will be added by default.

        :param kwargs: Items to add to the Dataset configuration.
        :return:
        """

        # Get items set as keyword arguments
        fields = tuple(kwargs.keys())
        args = tuple(kwargs.values())
        # Check if a dataset_config already exists (child class will have the parent's config by default)
        if 'dataset_config' in self.__dict__:
            # Get items set in the existing config
            for key, value in self.dataset_config._asdict().items():
                # Only new items are required for children, check if the parent's items are set again anyway
                if key not in fields:
                    fields += (key,)
                    args += (value,)
        # Create namedtuple with collected items
        return namedtuple('dataset_config', fields)._make(args)

    def create_dataset(self) -> BaseDataset:
        """
        Create an instance of dataset_class with given parameters.

        :return: BaseDataset object
        """

        try:
            dataset = self.dataset_class(config=self.dataset_config)
        except:
            raise ValueError(f"[{self.name}] Given dataset_class got an unexpected keyword argument 'config'")
        if not isinstance(dataset, BaseDataset):
            raise TypeError(f"[{self.name}] Wrong dataset_class type: BaseDataset required, get {self.dataset_class}")
        return dataset

    def __str__(self) -> str:
        """
        :return: String containing information about the BaseDatasetConfig object
        """
        # Todo: fields in Configs are the set in Managers or objects, the remove __str__ method
        description = "\n"
        description += f"{self.name}\n"
        description += f"    Dataset class: {self.dataset_class.__name__}\n"
        description += f"    Max size: {self.dataset_config.max_size}\n"
        description += f"    Dataset dir: {self.dataset_dir}\n"
        description += f"    Shuffle dataset: {self.shuffle_dataset}\n"
        return description
