"""
beamTrainingFC.py
Script used to train a FC network on FEM beam deformations
"""

import torch

from Sandbox.BeamConfig.BeamConfig import BeamConfig
from Sandbox.FEMBeam.FEMBeam import FEMBeam as Beam

from DeepPhysX_PyTorch.FC.FCConfig import FCConfig
from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX_Core.Pipelines.BaseTrainer import BaseTrainer



# ENVIRONMENT PARAMETERS
grid_resolution = [25, 5, 5]
grid_min = [0., 0., 0.]
grid_max = [100, 15, 15]
fixed_box = [0., 0., 0., 0., 15, 15]
p_grid = {'grid_resolution': grid_resolution, 'grid_min': grid_min, 'grid_max': grid_max, 'fixed_box': fixed_box}

# TRAINING PARAMETERS
nb_hidden_layers = 2
nb_node = grid_resolution[0] * grid_resolution[1] * grid_resolution[2]
layers_dim = [nb_node * 3] + [nb_node * 3 for _ in range(nb_hidden_layers + 1)] + [nb_node * 3]
nb_epoch = 10
nb_batch = 15
batch_size = 32


def createScene(root_node=None):
    # Environment config
    env_config = BeamConfig(environment_class=Beam, root_node=root_node, p_grid=p_grid, always_create_data=True)

    # Network config
    net_config = FCConfig(network_name="beam_FC", save_each_epoch=False,
                          loss=torch.nn.MSELoss, lr=1e-5, optimizer=torch.optim.Adam,
                          dim_output=3, dim_layers=layers_dim)

    # Dataset config
    dataset_config = BaseDatasetConfig(partition_size=1, shuffle_dataset=True)

    trainer = BaseTrainer(session_name="beam_FC_training_625", dataset_config=dataset_config,
                          environment_config=env_config, network_config=net_config,
                          nb_epochs=nb_epoch, nb_batches=nb_batch, batch_size=batch_size)
    trainer.execute()


if __name__ == '__main__':
    createScene()
