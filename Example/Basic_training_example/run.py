# Basic python imports
import sys
import torch

# DeepPhysX's Core imports
from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX_Core.Pipelines.BaseTrainer import BaseTrainer
from DeepPhysX.Example.Basic_training_example.Env import MeanEnvironment as Environment

# DeepPhysX's Sofa imports
from DeepPhysX_Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX_Core.AsyncSocket.BytesNumpyConverter import BytesNumpyConverter

# DeepPhysX's Pytorch imports
from DeepPhysX_PyTorch.FC.FCConfig import FCConfig


def createScene():
    env_config = BaseEnvironmentConfig(environment_class=Environment,
                                       visualizer_class=None,
                                       environment_file=sys.modules[Environment.__module__].__file__,
                                       number_of_thread=int(sys.argv[1]),
                                       socket_data_converter=BytesNumpyConverter,
                                       always_create_data=False)

    network_config = FCConfig(loss=torch.nn.MSELoss,
                              lr=1e-5,
                              optimizer=torch.optim.Adam,
                              dim_layers=[50, 50, 1],
                              dim_output=1)

    dataset_config = BaseDatasetConfig(partition_size=1, shuffle_dataset=True)

    trainer = BaseTrainer(session_name='training/example',
                          dataset_config=dataset_config,
                          environment_config=env_config,
                          network_config=network_config,
                          nb_epochs=40, nb_batches=1500, batch_size=10)
    trainer.execute()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: python3 {sys.argv[0]} <nb_thread>")
        sys.exit(1)
    createScene()
