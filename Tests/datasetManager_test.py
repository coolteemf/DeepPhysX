import os

import numpy as np

from DeepPhysX.Manager.Manager import Manager


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
    session_name = 'test_session'
    network_config = None
    max_size = 0.000001

    # Generate
    manager = Manager(session_name=session_name, network_config=network_config, trainer=True, manager_dir=None,
                      dataset_dir=None, network_dir=None, partition_size=max_size, shuffle_dataset=False,
                      save_each_epoch=False)
    dsm = manager.datasetManager
    for epoch in range(1):
        train_idx = 0
        test_idx = 0
        train_idx = generate(dsm, train_idx, 'train', epoch, 50)
        test_idx = generate(dsm, test_idx, 'test', epoch, 25)
        train_idx = generate(dsm, train_idx, 'train', epoch, 50)
        test_idx = generate(dsm, test_idx, 'test', epoch, 25)
    manager.close()

    # Read
    manager_dir = os.path.join(os.getcwd(), session_name)
    dataset_dir = os.path.join(manager_dir, 'dataset')
    manager = Manager(session_name=session_name, network_config=network_config, trainer=True, manager_dir=None,
                      dataset_dir=dataset_dir, network_dir=None, partition_size=max_size, shuffle_dataset=True,
                      save_each_epoch=False)
    dsm = manager.datasetManager
    for epoch in range(1):
        read(dsm, 'train', 50)
        print("")
        read(dsm, 'test', 25)
    manager.close()

    return


if __name__ == '__main__':
    main()
