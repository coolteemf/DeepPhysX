"""
prediction.py
Python script for DeepPhysX prediction pipeline on liver deformations with UNet.
Run with: 'python3 prediction.py'
"""

# Python imports
import os
import sys
import torch

# Sofa imports
import Sofa.Gui

# DeepPhysX Core imports
from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX_Core.Visualizer.MeshVisualizer import MeshVisualizer

# DeepPhysX Torch imports
from DeepPhysX_PyTorch.UNet.UNetConfig import UNetConfig

# DeepPhysX Sofa imports
from DeepPhysX_Sofa.Environment.SofaEnvironmentConfig import SofaEnvironmentConfig, BytesNumpyConverter
from DeepPhysX_Sofa.Runner.SofaRunner import SofaRunner

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
                                       environment_file=sys.modules[Livers.__module__].__file__,
                                       socket_data_converter=BytesNumpyConverter,
                                       always_create_data=False,
                                       use_prediction_in_environment=False,
                                       as_tcpip_client=False)

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

    # Runner
    runner = SofaRunner(session_name='training/liver',
                        dataset_config=dataset_config,
                        environment_config=env_config,
                        network_config=net_config,
                        visualizer_class=None,
                        session_dir=os.path.dirname(os.path.abspath(__file__)) + '/training/trained_liver',
                        nb_steps=0,
                        record_inputs=False,
                        record_outputs=False)
    return runner


if __name__ == '__main__':
    runner = main()
    Sofa.Gui.GUIManager.Init("main", "qglviewer")
    Sofa.Gui.GUIManager.createGUI(runner.root, __file__)
    Sofa.Gui.GUIManager.SetDimension(1080, 1080)
    Sofa.Gui.GUIManager.MainLoop(runner.root)
    Sofa.Gui.GUIManager.closeGUI()
    # Manually close the runner (security if stuff like additional dataset need to be saved)
    runner.close()
