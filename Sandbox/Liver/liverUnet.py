"""
liverUnet.py
Python script for both DeepPhysX training and prediction pipelines on liver deformation with UNet.
The default environment 'BothLiver' contains a FEM model and a NN model. Other scene can be added in the 'scene_map'
dictionary.
Run with:
    'python3 liverUnet.py -t' for training
    'python3 liverUnet.py x' for prediction with x being either a scene name either it's corresponding number in the
                             'scene_map' dictionary.
"""

# Import python library packages
import sys
import torch
import Sofa.Gui

# Import stuff to build the simulation
from Sandbox.Liver.Config.LiverConfig import LiverConfig
from Sandbox.Liver.Scene.BothLiver import BothLiver as Liver
from Sandbox.Liver.Config.parameters import *

# Import DeepPhysX packages
from DeepPhysX_PyTorch.UNet.UNetConfig import UNetConfig
from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX_Core.Pipelines.BaseTrainer import BaseTrainer
from DeepPhysX_Sofa.Runner.SofaRunner import SofaRunner
from DeepPhysX_Core.Visualizer.MeshVisualizer import MeshVisualizer


# Configure script
training = False
if len(sys.argv) > 1:
    # Check whether if the script is used for training (-t) or prediction (default)
    if sys.argv[1] == '-t':
        training = True
    # Training scene is default; if prediction, specify the chosen one
    else:
        scene_map = {'0': 'TrainingLiver'}
        if (sys.argv[1] in scene_map.keys()) or (sys.argv[1] in scene_map.values()):
            scene = scene_map[sys.argv[1]] if sys.argv[1] in scene_map.keys() else sys.argv[1]
            module = 'Sandbox.Liver.Scene.' + scene
            exec(f'from {module} import {scene} as Liver')


def createScene(root_node=None):
    """
    Automatically called when launching a Sofa scene or called from main to create the scene graph.

    :param root_node: Sofa.Core.Node() object.
    :return: root_node
    """
    # Environment config
    env_config = LiverConfig(environment_class=Liver, root_node=root_node, always_create_data=True,
                             visualizer_class=MeshVisualizer, p_liver=p_liver, p_grid=p_grid, p_force=p_force)
    # Network config
    net_config = UNetConfig(network_name="liver_UNet", save_each_epoch=False,
                            loss=torch.nn.MSELoss, lr=1e-6, optimizer=torch.optim.Adam,
                            steps=3, first_layer_channels=128, nb_classes=3,
                            nb_input_channels=3, nb_dims=3, border_mode='same', two_sublayers=True,
                            grid_shape=grid_resolution, data_scale=1000.)
    # Dataset config
    dataset_config = BaseDatasetConfig(partition_size=1, shuffle_dataset=True)

    # Training case
    if training:
        trainer = BaseTrainer(session_name="trainings/liver", dataset_config=dataset_config,
                              environment_config=env_config, network_config=net_config,
                              nb_epochs=100, nb_batches=30, batch_size=10)
        trainer.execute()
    # Prediction case
    else:
        man_dir = os.path.dirname(os.path.abspath(__file__)) + '/trainings/liver'
        runner = SofaRunner(session_name="session", dataset_config=dataset_config,
                            environment_config=env_config, network_config=net_config, session_dir=man_dir, nb_steps=0,
                            record_inputs=False, record_outputs=False)
        return runner


if __name__ == '__main__':
    # Training case : execute DeepPhysX pipeline of BaseTrainer
    if training:
        createScene()
    # Prediction case : launch Sofa GUI
    else:
        runner = createScene()
        # Launch the GUI
        Sofa.Gui.GUIManager.Init("main", "qglviewer")
        Sofa.Gui.GUIManager.createGUI(runner.root, __file__)
        Sofa.Gui.GUIManager.SetDimension(1080, 1080)
        Sofa.Gui.GUIManager.MainLoop(runner.root)
        Sofa.Gui.GUIManager.closeGUI()
        # Manually close the runner (security if stuff like additional dataset need to be saved)
        runner.close()
