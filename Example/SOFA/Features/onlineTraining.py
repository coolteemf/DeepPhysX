"""
onlineTraining.py
Run the pipeline BaseTrainer to create a new Dataset while training the Network.
"""

# Python related imports
from torch.nn import MSELoss
from torch.optim import Adam

# DeepPhysX related imports
from DeepPhysX_Core.Pipelines.BaseTrainer import BaseTrainer
from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX_Core.Visualizer.VedoVisualizer import VedoVisualizer
from DeepPhysX_Sofa.Environment.SofaEnvironmentConfig import SofaEnvironmentConfig
from DeepPhysX_PyTorch.FC.FCConfig import FCConfig

# Session imports
from Environment.EnvironmentTraining import MeanEnvironmentTraining


def launch_training():
    # Define the number of points and the dimension
    nb_points = 30
    dimension = 3
    # Environment configuration
    environment_config = SofaEnvironmentConfig(environment_class=MeanEnvironmentTraining,
                                               visualizer=VedoVisualizer,
                                               param_dict={'constant': False,
                                                           'data_size': [nb_points, dimension],
                                                           'sleep': False},
                                               as_tcp_ip_client=True,
                                               number_of_thread=4)
    # Fully Connected configuration (the number of neurones on the first and last layer is defined by the total amount
    # of parameters in the input and the output vectors respectively)
    network_config = FCConfig(loss=MSELoss,
                              lr=1e-3,
                              optimizer=Adam,
                              dim_layers=[nb_points * dimension, nb_points * dimension, dimension],
                              dim_output=dimension)
    # Dataset configuration
    dataset_config = BaseDatasetConfig(shuffle_dataset=True,
                                       normalize=False)
    # Create DataGenerator
    trainer = BaseTrainer(session_dir='sessions',
                          session_name='online_training',
                          environment_config=environment_config,
                          dataset_config=dataset_config,
                          network_config=network_config,
                          nb_epochs=5,
                          nb_batches=100,
                          batch_size=10)
    # Launch the training session
    trainer.execute()


if __name__ == '__main__':
    launch_training()
