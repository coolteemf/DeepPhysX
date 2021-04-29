import os

import numpy as np

from DeepPhysX.Dataset.BaseDataset import Dataset
from DeepPhysX.Dataset.BaseDatasetConfig import DatasetConfig
from DeepPhysX.Manager.DatasetManager import DatasetManager


def generate(dsm, idx, mode, epoch, t):
    dsm.setMode(mode)
    for _ in range(t):
        data = {'in': np.array([[epoch + 1, idx + 1]]), 'out': np.array([[[epoch + 1, idx + 1], [0, 0]]])}
        dsm.addData(data)
        idx += 1
    return idx


def main():

    # Creating a dataset with multiple partitions on different modes
    create_dataset_config = DatasetConfig(dataset_class=Dataset,
                                          partition_size=0.000001,
                                          dataset_dir=None,
                                          generate_data=True,
                                          shuffle_dataset=True)
    create_dataset_manager = DatasetManager(session_name='TestSession',
                                            dataset_config=create_dataset_config,
                                            manager_dir=os.path.join(os.getcwd(), 'datasetManager/create/'),
                                            trainer=True)
    print(create_dataset_manager.getDescription())
    for epoch in range(2):
        train_idx, test_idx = 0, 0
        train_idx = generate(create_dataset_manager, train_idx, 'train', epoch, 50)
        test_idx = generate(create_dataset_manager, test_idx, 'test', epoch, 25)
    dataset_dir = create_dataset_manager.datasetDir
    create_dataset_manager.close()

    # Loading an existing dataset
    load_dataset_config = DatasetConfig(dataset_class=Dataset,
                                        partition_size=1,
                                        dataset_dir=dataset_dir,
                                        generate_data=False,
                                        shuffle_dataset=True)
    load_dataset_manager = DatasetManager(session_name='TestSession',
                                          dataset_config=load_dataset_config,
                                          manager_dir=os.path.join(os.getcwd(), 'datasetManager/load/'),
                                          trainer=True)
    for _ in range(5):
        sample = load_dataset_manager.getNextSample(batched=True)
        print("Input: {}, Output: {}".format(sample['in'], sample['out']))
        sample = load_dataset_manager.getNextSample(batched=False)
        print("Input: {}, Output: {}".format(sample['in'], sample['out']))
    batch = load_dataset_manager.getNextBatch(batch_size=4)
    print("Input: {}, Output: {}".format(batch['in'], batch['out']))
    load_dataset_manager.close()

    return


if __name__ == '__main__':
    main()
