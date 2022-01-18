# Basic python imports
import sys
import torch

# DeepPhysX's Core imports
from DeepPhysX_Core import BaseDatasetConfig
from DeepPhysX_Core import BaseTrainer
from DeepPhysX_Core import BaseEnvironmentConfig
from DeepPhysX_Core import VedoVisualizer

from DeepPhysX.Example.Mean_training_visualizer.Producer import MeanEnvironment as Environment

# DeepPhysX's Pytorch imports
from DeepPhysX_PyTorch import FCConfig



def createScene():
    env_config = BaseEnvironmentConfig(environment_class=Environment,                                   # Environment class to launch in external process
                                       number_of_thread=int(sys.argv[1]),                               # Number of threads/process to launch
                                       visualizer=VedoVisualizer)                                       # Provide a way to visualize data

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
                          nb_epochs=3, nb_batches=10, batch_size=1)                                     # Training settings

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
    elif len(sys.argv) == 1:
        print(f"Number of process not specified will run a single client")
        sys.argv.append(1)
    createScene()
