# Python imports
import sys

# DeepPhysX Core imports
from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX_Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX_Core.Network.BaseNetworkConfig import BaseNetworkConfig
from DeepPhysX_Core.Pipelines.BaseTrainer import BaseTrainer

# Working session imports
from Environment import Environment
from DummyNetwork import DummyNetwork, DummyOptimizer


def main():
    # Environment config
    env_config = BaseEnvironmentConfig(environment_class=Environment,
                                       environment_file=sys.modules[Environment.__module__].__file__,
                                       number_of_thread=1,
                                       as_tcpip_client=True,
                                       use_prediction_in_environment=True,
                                       port=10002)

    # Network config
    net_config = BaseNetworkConfig(network_class=DummyNetwork,
                                   optimization_class=DummyOptimizer,
                                   network_name='DummyNetwork',
                                   network_type='Dummy',
                                   which_network=0,
                                   save_each_epoch=False,
                                   loss=lambda x: x,
                                   lr=0.01,
                                   optimizer=object)

    # Dataset config
    dataset_config = BaseDatasetConfig(partition_size=0.000005,
                                       shuffle_dataset=True)

    # Trainer
    trainer = BaseTrainer(session_name="sessions/dummy",
                          dataset_config=dataset_config,
                          environment_config=env_config,
                          network_config=net_config,
                          nb_epochs=2,
                          nb_batches=10,
                          batch_size=2)
    trainer.execute()


if __name__ == '__main__':
    main()
