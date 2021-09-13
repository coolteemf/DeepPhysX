# Basic python imports
import sys
import torch

# DeepPhysX's Core imports
from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX_Core.Visualizer.MeshVisualizer import MeshVisualizer
from DeepPhysX_Core.Pipelines.BaseTrainer import BaseTrainer
from Example.Training_example.EnvironmentSofa import FEMBeam

# DeepPhysX's Sofa imports
from DeepPhysX_Sofa.Environment.SofaEnvironmentConfig import SofaEnvironmentConfig, BytesNumpyConverter

# DeepPhysX's Pytorch imports
from DeepPhysX_PyTorch.FC.FCConfig import FCConfig


def createScene(root_node=None):
    env_config = SofaEnvironmentConfig(environment_class=FEMBeam,
                                       environment_file=sys.modules[FEMBeam.__module__].__file__,
                                       number_of_thread=int(sys.argv[1]),
                                       socket_data_converter=BytesNumpyConverter,
                                       always_create_data=False)

    neurones_per_layer = 25*5*5*3  # Grid of size 25x5x5
    network_config = FCConfig(loss=torch.nn.MSELoss,
                              lr=1e-5,
                              optimizer=torch.optim.Adam,
                              dim_layers=[neurones_per_layer, neurones_per_layer, neurones_per_layer],
                              dim_output=3)

    dataset_config = BaseDatasetConfig(partition_size=1, shuffle_dataset=True)

    trainer = BaseTrainer(session_name='training/example',
                          dataset_config=dataset_config,
                          environment_config=env_config,
                          network_config=network_config,
                          visualizer_class=MeshVisualizer,
                          nb_epochs=40, nb_batches=100, batch_size=5)
    trainer.execute()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: python3 {sys.argv[0]} <nb_thread>")
        sys.exit(1)
    createScene()
