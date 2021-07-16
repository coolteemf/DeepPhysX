"""
beamPredictionFC.py
Run this script to see the predictions of a trained network in the dedicated Sofa scene. Only run scenes from the
repository 'NNBeam'.
The default scene is NNBeam.py, but other scene can be launched using 'python3 beamPredictionFC.py x' with x being
either the name of the scene either it's corresponding number in the following scene map dictionary.
"""

import os
import sys
sys.path.append(os.getcwd())
import torch
import Sofa.Gui

from DeepPhysX_PyTorch.FC.FCConfig import FCConfig
from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX_Sofa.Runner.SofaRunner import SofaRunner
from DeepPhysX_Core.Visualizer.MeshVisualizer import MeshVisualizer

from Sandbox.Beam.BeamConfig.BeamConfig import BeamConfig
from Sandbox.Beam.NNBeam.NNBeam import NNBeam as Beam

if len(sys.argv) > 1:
    scene_map = {'0': 'NNBeam', '1': 'NNBeamCompare', '2': 'NNBeamMouse', '3': 'NNBeamContact', '4': 'NNBeamCollision',
                 '5': 'NNBeamCompareConstant', '6': 'NNBeamDynamic'}
    arg = '0' if (sys.argv[1] not in scene_map.keys()) and (sys.argv[1] not in scene_map.values()) else sys.argv[1]
    scene = scene_map[arg] if (arg in scene_map.keys()) else arg
    scene_rep = 'Sandbox.NNBeam.' + scene
    exec(f'from {scene_rep} import {scene} as Beam')


# ENVIRONMENT PARAMETERS
grid_resolution = [25, 5, 5]    # [40, 10, 10]
grid_min = [0., 0., 0.]
grid_max = [100, 15, 15]       # [100., 25., 25.]
fixed_box = [0., 0., 0., 0., 25., 25.]

free_box = [49.5, -0.5, -0.5, 100.5, 25.5, 25.5]
top_box = [49.5, 24.5, -0.5, 100.5, 25.5, 25.5]     # [89.5, 14.5, -0.5, 100.5, 15.5, 15.5]
all_box = [0., 0., 0., 100.5, 25.5, 25.5]

p_grid = {'grid_resolution': grid_resolution, 'grid_min': grid_min, 'grid_max': grid_max,
          'fixed_box': fixed_box, 'free_box': free_box, 'all_box': all_box, 'top_box': top_box}

# NETWORK PARAMETERS
nb_hidden_layers = 2
nb_node = grid_resolution[0] * grid_resolution[1] * grid_resolution[2]
layers_dim = [nb_node * 3] + [nb_node * 3 for _ in range(nb_hidden_layers + 1)] + [nb_node * 3]


def createScene(root_node=None):
    """
    Automatically called when launching a Sofa scene or called from main to create the scene graph.
    :param root_node: Sofa.Core.Node() object.
    :return: root_node
    """
    # Environment config
    env_config = BeamConfig(environment_class=Beam, root_node=root_node, p_grid=p_grid, always_create_data=True,
                            visualizer_class=None)
    # Network config
    net_config = FCConfig(network_name="beam_FC", save_each_epoch=True,
                          loss=torch.nn.MSELoss, lr=1e-5, optimizer=torch.optim.Adam,
                          dim_output=3, dim_layers=layers_dim)
    # Dataset config
    dataset_config = BaseDatasetConfig(partition_size=1, shuffle_dataset=True)
    # Runner
    man_dir = os.path.dirname(os.path.realpath(__file__)) + '/trainings/beam_FC_625'
    runner = SofaRunner(session_name="session", dataset_config=dataset_config,
                        environment_config=env_config, network_config=net_config, session_dir=man_dir, nb_steps=0,
                        record_inputs=False, record_outputs=False)
    return runner


# Executed through python interpreter
if __name__ == '__main__':
    # Create scene graph
    runner = createScene()
    # Launch the GUI
    Sofa.Gui.GUIManager.Init("main", "qglviewer")
    Sofa.Gui.GUIManager.createGUI(runner.root, __file__)
    Sofa.Gui.GUIManager.SetDimension(1080, 1080)
    Sofa.Gui.GUIManager.MainLoop(runner.root)
    Sofa.Gui.GUIManager.closeGUI()
    # Manually close the runner (security if stuff like additional dataset need to be saved)
    runner.close()
