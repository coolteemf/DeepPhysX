# Basic python imports
import sys
import torch

# DeepPhysX's Core imports
from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX_Core.Visualizer.MeshVisualizer import MeshVisualizer
from DeepPhysX_Core.Pipelines.BaseTrainer import BaseTrainer

# DeepPhysX's Sofa imports
from DeepPhysX_Sofa.Environment.SofaEnvironmentConfig import SofaEnvironmentConfig, BytesNumpyConverter

# DeepPhysX's Pytorch imports
from DeepPhysX_PyTorch.FC.FCConfig import FCConfig

# Trainin session relative imports
from Sessions.Beam.Beam import FEMBeam, grid_dofs_count
from Sessions.Beam.PhysicBasedOptimization import PhysicBasedOptimization


def createScene(root_node=None):
    env_config = SofaEnvironmentConfig(environment_class=FEMBeam,
                                       environment_file=sys.modules[FEMBeam.__module__].__file__,
                                       number_of_thread=int(sys.argv[1]),
                                       socket_data_converter=BytesNumpyConverter,
                                       always_create_data=False,
                                       use_prediction_in_environment=True)

    network_config = FCConfig(optimization_class=PhysicBasedOptimization,
                              loss=torch.nn.MSELoss,
                              lr=1e-4,
                              optimizer=torch.optim.Adam,
                              dim_layers=[grid_dofs_count, grid_dofs_count, grid_dofs_count],
                              dim_output=3)

    dataset_config = BaseDatasetConfig(partition_size=1, shuffle_dataset=True)

    trainer = BaseTrainer(session_name='training/beamFC',
                          dataset_config=dataset_config,
                          environment_config=env_config,
                          network_config=network_config,
                          visualizer_class=MeshVisualizer,
                          nb_epochs=50, nb_batches=20, batch_size=5)
    trainer.execute()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: python3 {sys.argv[0]} <nb_thread>")
        sys.exit(1)
    createScene()
