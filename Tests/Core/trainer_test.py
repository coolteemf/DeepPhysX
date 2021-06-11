from numpy_environment import NumpyEnvironment
from numpy_Network import NumpyNetwork, NumpyOptimisation
from DeepPhysX.Pipelines.BaseTrainer import BaseTrainer
from DeepPhysX.Dataset.BaseDataset import BaseDataset
from DeepPhysX.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX.Network.BaseNetworkConfig import BaseNetworkConfig
from DeepPhysX.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig


def main():

    epochs = 100
    nb_batches = 50
    batch_size = 10
    lr = 0.005
    partition_size = 0.01

    dataset_config = BaseDatasetConfig(dataset_class=BaseDataset, dataset_dir=None, partition_size=partition_size,
                                       shuffle_dataset=True)
    network_config = BaseNetworkConfig(network_class=NumpyNetwork, optimization_class=NumpyOptimisation,
                                       network_dir=None, save_each_epoch=False, lr=lr)
    network_config.training_stuff = True
    environment_config = BaseEnvironmentConfig(environment_class=NumpyEnvironment, simulations_per_step=1,
                                               always_create_data=True)
    trainer = BaseTrainer(network_config=network_config, dataset_config=dataset_config,
                          environment_config=environment_config, session_name="Session(1)", new_session=False,
                          nb_epochs=1, nb_batches=1, batch_size=10)
    trainer.execute()

    return


if __name__ == '__main__':
    main()
