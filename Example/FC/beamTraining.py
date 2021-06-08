from torch.optim import Adam
from torch.nn import MSELoss

from BeamEnvironmentConfig import BeamEnvironmentConfig
from DeepPhysX_PyTorch.FC.FCConfig import FCConfig
from DeepPhysX.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX.Pipelines.BaseTrainer import BaseTrainer

# ENVIRONMENT PARAMETERS
grid_resolution = [16, 4, 4]
grid_min = [0., 0., 0.]
grid_max = [1.0, 0.2, 0.2]
fixed_box = [0., 0., 0., 0., 0.25, 0.25]
p_grid = {'grid_resolution': grid_resolution,
          'min': grid_min,
          'max': grid_max,
          'fixed_box': fixed_box}

# TRAINING PARAMETERS
nb_hidden_layers = 2
nb_dof = grid_resolution[0] * grid_resolution[1] * grid_resolution[2]
layers_dim = [nb_dof * 3] + [nb_dof * 3 for _ in range(nb_hidden_layers + 1)] + [nb_dof * 3]
nb_epoch = 100
nb_batch = 20
batch_size = 10


def createScene(root_node=None, runSofa=True):
    # Environment config
    env_config = BeamEnvironmentConfig(root_node=root_node, p_grid=p_grid,
                                       always_create_data=True)

    # Network config
    net_config = FCConfig(network_name="beam_FC", save_each_epoch=True,
                          loss=MSELoss, lr=1e-5, optimizer=Adam,
                          dim_output=3, dim_layers=layers_dim)

    # Dataset config
    dataset_config = BaseDatasetConfig()
    BaseDatasetConfig(partition_size=1, generate_data=True, shuffle_dataset=True)

    if not runSofa:
        trainer = BaseTrainer(session_name="beam_FC_training", dataset_config=dataset_config, environment_config=env_config,
                              network_config=net_config, nb_epochs=nb_epoch, nb_batches=nb_batch, batch_size=batch_size)
        trainer.execute()
    else:
        env_config.createEnvironment(training=False)


if __name__ == '__main__':
    createScene(runSofa=False)
