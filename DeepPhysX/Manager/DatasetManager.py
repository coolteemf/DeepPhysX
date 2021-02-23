import os

from DeepPhysX.Dataset.Dataset import Dataset


class DatasetManager:
    def __init__(self, dataset_dir, network_name, partition_size=1, read_only=False):
        self.hereAbsolutePath = '{}/'.format(os.path.dirname(os.path.realpath(__file__)))
        
        print(self.hereAbsolutePath)
