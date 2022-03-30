"""
training.py
Launch the training session with a VedoVisualizer.
"""

# Python related imports
import os
import sys
import torch

# DeepPhysX related imports
from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX_Core.Visualizer.VedoVisualizer import VedoVisualizer
from DeepPhysX_Core.Pipelines.BaseTrainer import BaseTrainer
from DeepPhysX_PyTorch.FC.FCConfig import FCConfig
from DeepPhysX_Sofa.Environment.SofaEnvironmentConfig import SofaEnvironmentConfig

# Working session imports
from Environment.ArmadilloTraining import ArmadilloTraining
import Environment.parameters as parameters

# Training parameters
nb_epochs = 400
nb_batch = 200
batch_size = 32
lr = 1e-5


def launch_trainer():

    # Environment config
    env_config = SofaEnvironmentConfig(environment_class=ArmadilloTraining,
                                       visualizer=VedoVisualizer,
                                       number_of_thread=int(sys.argv[1]),
                                       always_create_data=False)

    # UNet config
    nb_hidden_layers = 2
    nb_neurons = parameters.p_model.nb_nodes * 3
    layers_dim = [nb_neurons] + [nb_neurons for _ in range(nb_hidden_layers + 1)] + [nb_neurons]
    net_config = FCConfig(network_name='armadillo_FC',
                          loss=torch.nn.MSELoss,
                          lr=lr,
                          optimizer=torch.optim.Adam,
                          dim_output=3,
                          dim_layers=layers_dim)

    # Dataset config
    dataset_config = BaseDatasetConfig(partition_size=1, shuffle_dataset=True)

    # Trainer
    trainer = BaseTrainer(session_dir=os.getcwd(),
                          session_name="sessions/armadillo",
                          dataset_config=dataset_config,
                          environment_config=env_config,
                          network_config=net_config,
                          nb_epochs=nb_epochs,
                          nb_batches=nb_batch,
                          batch_size=batch_size)
    trainer.execute()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: python3 {sys.argv[0]} <nb_thread>")
        sys.exit(1)
    launch_trainer()
