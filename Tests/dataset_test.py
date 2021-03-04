"""
Test script for the DeepPhysX Dataset
"""

import numpy as np
import random
import os

try:
    from DeepPhysX.Manager.DatasetManager import DatasetManager
except ImportError:
    print("Cannot import DeepPhysX package")



def createData(size):
    data = np.empty((size, 10), dtype=np.float32)
    labels = np.empty(size, dtype=np.longlong)
    for i in range(size):
        val = round(random.random(), 2)
        while val >= 1.0:
            val = round(random.random(), 2)
        data[i] = np.array([val for _ in range(10)])
        labels[i] = int(10.0 * val)
    return data, labels


def main():
    x, y = createData(10)
    network_name = 'balec'
    max_size = 0.01
    dsm = DatasetManager(network_name=network_name, partition_size=max_size, read_only=False)
    print(x, y)

    return


if __name__ == '__main__':
    main()
