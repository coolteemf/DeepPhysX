"""
Online Training
Run the pipeline BaseTrainer to create a new Dataset while training the Network.
"""

# Python related imports
import os
from torch.nn import MSELoss
from torch.optim import Adam

# DeepPhysX related imports
from DeepPhysX_Core.Pipelines.BaseTrainer import BaseTrainer
from DeepPhysX_Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX_PyTorch.FC.FCConfig import FCConfig
from DeepPhysX_Core.Visualizer.VedoVisualizer import VedoVisualizer

# Session imports
from Environment import MeanEnvironment


def main():
    # Define the number of points and the dimension
    nb_points = 30
    dimension = 3
    # Environment configuration
    environment_config = BaseEnvironmentConfig(environment_class=MeanEnvironment,
                                               visualizer=VedoVisualizer,
                                               param_dict={'nb_points': nb_points,
                                                           'dimension': dimension},
                                               as_tcp_ip_client=True,
                                               number_of_thread=4)
    # Fully Connected configuration (the number of neurones on the first and last layer is defined by the total amount
    # of parameters in the input and the output vectors respectively)
    network_config = FCConfig(loss=MSELoss,
                              lr=1e-3,
                              optimizer=Adam,
                              dim_layers=[nb_points * dimension, nb_points * dimension, dimension],
                              dim_output=dimension)
    # Dataset configuration with the path to the existing Dataset
    dataset_config = BaseDatasetConfig(shuffle_dataset=True)
    # Create DataGenerator
    trainer = BaseTrainer(session_name='sessions/online_training',
                          environment_config=environment_config,
                          dataset_config=dataset_config,
                          network_config=network_config,
                          nb_epochs=5,
                          nb_batches=100,
                          batch_size=10)
    # Launch the training session
    trainer.execute()


if __name__ == '__main__':
    main()
