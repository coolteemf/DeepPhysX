"""
Test script for the DeepPhysX Dataset
"""

import numpy as np
import random
import os

from DeepPhysX.Manager.DatasetManager import DatasetManager


def createData(n, m):
    data = np.array([n, m])
    output = np.random.random(2)
    return data, output


def generate(dsm, idx, mode, epoch, t):
    dsm.setMode(mode)
    for _ in range(t):
        x, y = createData(epoch, idx)
        dsm.addData(x, y)
        idx += 1
    return idx


def read(dsm, mode, t):
    dsm.setMode(mode)
    for _ in range(t):
        sample = dsm.getNextSample(batched=True)
        print(sample)


def main():
    network_name = 'network'
    max_size = 0.000001
    # Generate
    dsm = DatasetManager(network_name=network_name, partition_size=max_size, mode='train', generate_data=True)
    for epoch in range(1):
        train_idx = 0
        test_idx = 0
        train_idx = generate(dsm, train_idx, 'train', epoch, 50)
        test_idx = generate(dsm, test_idx, 'test', epoch, 25)
        train_idx = generate(dsm, train_idx, 'train', epoch, 50)
        test_idx = generate(dsm, test_idx, 'test', epoch, 25)
    dsm.close()
    # Read
    dsm = DatasetManager(network_name=network_name, partition_size=max_size, mode='train', generate_data=False,
                         shuffle_dataset=True)
    for epoch in range(1):
        read(dsm, 'train', 50)
        print("")
        read(dsm, 'test', 25)
        """read(dsm, 'train', 50)
        read(dsm, 'test', 25)"""
    dsm.close()
    return


if __name__ == '__main__':
    main()
