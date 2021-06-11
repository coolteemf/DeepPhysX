from numpy_environment import NumpyEnvironment
from numpy_Network import NumpyNetwork, NumpyOptimisation
from DeepPhysX.Pipelines.BaseRunner import BaseRunner
from DeepPhysX.Dataset.BaseDataset import BaseDataset
from DeepPhysX.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX.Network.BaseNetworkConfig import BaseNetworkConfig
from DeepPhysX.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig


def main():

    partition_size = 0.01

    dataset_config = BaseDatasetConfig(dataset_class=BaseDataset, dataset_dir=None, partition_size=partition_size,
                                       shuffle_dataset=True)
    network_config = BaseNetworkConfig(network_class=NumpyNetwork, optimization_class=NumpyOptimisation,
                                       network_dir=None, save_each_epoch=False)
    environment_config = BaseEnvironmentConfig(environment_class=NumpyEnvironment, simulations_per_step=1,
                                               always_create_data=True)
    trainer = BaseRunner(network_config=network_config, dataset_config=dataset_config,
                         environment_config=environment_config, session_name="Session(1)",
                         nb_steps=10, record_inputs=True, record_outputs=True)
    trainer.execute()
    return


if __name__ == '__main__':
    main()
