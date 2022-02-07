# Python imports
import os
import sys
import shutil

# DeepPhysX Core imports
from DeepPhysX_Core.Pipelines.BaseDataGenerator import BaseDataGenerator
from DeepPhysX_Core.Pipelines.BaseTrainer import BaseTrainer
from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX_Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX_Core.Network.BaseNetworkConfig import BaseNetworkConfig

# Working session imports
from Environment import Environment
from DummyNetwork import DummyNetwork, DummyOptimizer


def generate_dataset(nb_batch, batch_size):
    env_config = BaseEnvironmentConfig(environment_class=Environment,
                                       environment_file=sys.modules[Environment.__module__].__file__)
    dataset_config = BaseDatasetConfig(partition_size=0.000001)
    pipeline = BaseDataGenerator(dataset_config=dataset_config,
                                 environment_config=env_config,
                                 session_name='generation',
                                 nb_batches=nb_batch,
                                 batch_size=batch_size)
    pipeline.execute()


def main(nb_batch, batch_size):
    env_config = BaseEnvironmentConfig(environment_class=Environment,
                                       environment_file=sys.modules[Environment.__module__].__file__,
                                       as_tcp_ip_client=True,
                                       use_prediction_in_environment=True)
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
    dataset_config = BaseDatasetConfig(dataset_dir=os.getcwd()+'/generation/dataset',
                                       shuffle_dataset=False)

    # Trainer
    trainer = BaseTrainer(session_name="training",
                          environment_config=env_config,
                          dataset_config=dataset_config,
                          network_config=net_config,
                          nb_epochs=2,
                          nb_batches=nb_batch,
                          batch_size=batch_size)
    trainer.execute()


if __name__ == '__main__':

    if os.path.exists(os.getcwd() + '/generation'):
        shutil.rmtree(os.getcwd() + '/generation')
    print("GENERATING DATASET...")
    generate_dataset(5, 10)

    if os.path.exists(os.getcwd() + '/training'):
        shutil.rmtree(os.getcwd() + '/training')
    print("TRAINING FROM EXISTING DATASET...")
    main(5, 10)
