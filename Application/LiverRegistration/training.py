"""
training.py
Python script for DeepPhysX training pipeline on liver deformations with UNet.
Run with: 'python3 training.py <nb_thread>'
"""

# Python imports
import sys
import torch

# DeepPhysX Core imports
from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX_Core.Visualizer.MeshVisualizer import MeshVisualizer
from DeepPhysX_Core.Pipelines.BaseTrainer import BaseTrainer

# DeepPhysX Torch imports
from DeepPhysX_PyTorch.UNet.UNetConfig import UNetConfig

# DeepPhysX Sofa imports
from DeepPhysX_Sofa.Environment.SofaEnvironmentConfig import SofaEnvironmentConfig, BytesNumpyConverter

# Working session imports
from Application.LiverRegistration.SofaScene.Livers import Livers
import Application.LiverRegistration.SofaScene.parameters as parameters


def main():
    """
    Launch DeepPhysX training.

    :return:
    """

    # Liver scene config
    env_config = SofaEnvironmentConfig(environment_class=Livers,
                                       visualizer_class=MeshVisualizer,
                                       environment_file=sys.modules[Livers.__module__].__file__,
                                       number_of_thread=int(sys.argv[1]),
                                       socket_data_converter=BytesNumpyConverter,
                                       always_create_data=False,
                                       use_prediction_in_environment=False)

    # UNet config
    net_config = UNetConfig(network_name="liver_UNet",
                            save_each_epoch=False,
                            loss=torch.nn.MSELoss,
                            lr=parameters.lr,
                            optimizer=torch.optim.Adam,
                            steps=3,
                            first_layer_channels=128,
                            nb_classes=3,
                            nb_input_channels=1,
                            nb_dims=3,
                            border_mode='same',
                            two_sublayers=True,
                            grid_shape=parameters.grid_resolution,
                            data_scale=1000.)

    # Dataset config
    dataset_config = BaseDatasetConfig(partition_size=1, shuffle_dataset=True)

    # Trainer
    trainer = BaseTrainer(session_name="training/liver",
                          dataset_config=dataset_config,
                          environment_config=env_config,
                          network_config=net_config,

                          nb_epochs=parameters.nb_epochs,
                          nb_batches=parameters.nb_batch,
                          batch_size=parameters.batch_size)
    trainer.execute()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: python3 {sys.argv[0]} <nb_thread>")
        sys.exit(1)
    main()
