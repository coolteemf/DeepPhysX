import numpy as np

from DeepPhysX.Dataset.BaseDatasetConfig import BaseDatasetConfig


def main():

    # Dataset configuration
    dataset_config = BaseDatasetConfig(partition_size=1)
    print(dataset_config.getDescription())

    # New dataset: dataset has the following format: {'in': [[in0], [in1], [in2]], 'out': [[out0], [out1], [out2]]}
    # Create dataset
    new_dataset = dataset_config.createDataset()
    for epoch in range(2):
        for iteration in range(2):
            inputs, outputs = np.array([[epoch + 1, iteration + 1]]), np.array([[[epoch + 1, iteration + 1], [0, 0]]])
            new_dataset.add(inputs, outputs)
    print(new_dataset.getDescription())
    print("Actual dataset size:", new_dataset.getSize())
    print("Initial Inputs:\n{}".format(new_dataset.data['in']))
    print("Initial Outputs:\n{}\n".format(new_dataset.data['out']))
    # Shuffle dataset
    new_dataset.shuffle()
    print("Shuffled Inputs:\n{}".format(new_dataset.data['in']))
    print("Shuffled Outputs:\n{}\n".format(new_dataset.data['out']))
    # Reset dataset
    inputs = new_dataset.data['in']
    outputs = new_dataset.data['out']
    new_dataset.reset()
    print("Reset dataset: {}\n".format(new_dataset.data))

    # Loading existing dataset
    loading_dataset = dataset_config.createDataset()
    loading_dataset.loadData(inputs[0:2], outputs[0:2])
    loading_dataset.loadData([inputs[2]], [outputs[2]])
    print("Loaded Inputs:\n{}".format(loading_dataset.data['in']))
    print("Loaded Outputs:\n{}\n".format(loading_dataset.data['out']))
    return


if __name__ == '__main__':
    main()
