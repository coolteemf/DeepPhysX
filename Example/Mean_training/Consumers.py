# Basic python imports
import sys
import torch

# DeepPhysX's Core imports
from DeepPhysX_Core import BaseDatasetConfig
from DeepPhysX_Core import BaseTrainer
from DeepPhysX.Example.Mean_training.Producer import MeanEnvironment as Environment

# DeepPhysX's Sofa imports
from DeepPhysX_Core import BaseEnvironmentConfig
from DeepPhysX_Core import BytesNumpyConverter

# DeepPhysX's Pytorch imports
from DeepPhysX_PyTorch import FCConfig


def createScene():
    env_config = BaseEnvironmentConfig(environment_class=Environment,                                   # Environment class to launch in external process
                                       environment_file=sys.modules[Environment.__module__].__file__,   # File containing this environment
                                       number_of_thread=int(sys.argv[1]),                               # Number of threads/process to launch
                                       socket_data_converter=BytesNumpyConverter)                       # How to convert data to/from TCPIP format

    # The number of neurones on the first and last layer is entierly
    # defined by the total amount of parameters in respectively the
    # input and the output
    network_config = FCConfig(loss=torch.nn.MSELoss,                                                    # Loss function associated with the learning
                              lr=1e-5,                                                                  # Learning rate
                              optimizer=torch.optim.Adam,                                               # Optimizer associated with the learning process
                              dim_layers=[50, 50, 2],                                                   # Number of neurones for each layer (in the case of fully connected)
                              dim_output=2)                                                             # Width of the output

    dataset_config = BaseDatasetConfig(partition_size=1)                                                # Max file size in Gb

    trainer = BaseTrainer(session_name='training/example',                                              # Name and relative location of the folder containing all of the data
                          dataset_config=dataset_config,                                                # Previously defined dataset configuration
                          environment_config=env_config,                                                # Previously defined environment configuration
                          network_config=network_config,                                                # Previously defined network configuration
                          nb_epochs=40, nb_batches=1500, batch_size=10)                                 # Training settings

    ##############################################################################
    #                                                                            #
    #                       EXECUTE THE TRAINING PROCESS                         #
    #                                                                            #
    ##############################################################################
    trainer.execute()


if __name__ == '__main__':
    if len(sys.argv) > 2:
        print(f"Usage: python3 {sys.argv[0]} <nb_thread>")
        sys.exit(1)
    else:
        print(f"Number of process not specified will run a single client")
        sys.argv.append(1)
    createScene()
