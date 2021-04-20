from MyEnvironment import MyEnvironment
from MyNetwork import MyNetwork, MyNetworkOptimisation
from DeepPhysX.Trainer.BaseTrainer import BaseTrainer
from DeepPhysX.Dataset.Dataset import Dataset
from DeepPhysX.Dataset.DatasetConfig import DatasetConfig
from DeepPhysX.Network.NetworkConfig import NetworkConfig
from DeepPhysX.Environment.EnvironmentConfig import EnvironmentConfig


def main():

    epochs = 1000
    nb_batches = 50
    batch_size = 10
    lr = 0.005
    partition_size = 0.01

    dataset_config = DatasetConfig(partition_size=partition_size, dataset_dir=None, generate_data=True,
                                   shuffle_dataset=True, dataset_class=Dataset)
    network_config = NetworkConfig(network_class=MyNetwork,
                                   optimization_class=MyNetworkOptimisation,
                                   network_name="myNetwork",
                                   lr=1e-6,
                                   network_dir=None,
                                   save_each_epoch=False)
    network_config.trainingMaterials = True

    # Train with single processing
    single_environment_config = EnvironmentConfig(environment_class=MyEnvironment,
                                                  simulations_per_step=1,
                                                  always_create_data=True,
                                                  multiprocessing=1,
                                                  multiprocess_method=None)
    trainer = BaseTrainer(session_name="trainer_test",
                          nb_epochs=epochs,
                          nb_batches=nb_batches,
                          batch_size=batch_size,
                          network_config=network_config,
                          dataset_config=dataset_config,
                          environment_config=single_environment_config)

    trainer.execute()

    return


if __name__ == '__main__':
    main()
