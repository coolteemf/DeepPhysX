"""
runSofa.py
Run this script to check if the Sofa scene is working well. Outside of DeepPhysX_Core pipeline, only run scenes in this
repository.
The default scene is FEMBeam.py, but other scene can be launched using 'python3 runSofa.py x' with x being either the
name of the scene either it's corresponding number in the following scene map dictionary.
"""


import os
import sys
sys.path.append(os.getcwd())
import Sofa.Gui

from DeepPhysX_Core.Visualizer.MeshVisualizer import MeshVisualizer

from Sandbox.Beam.BeamConfig.BeamConfig import BeamConfig
from Sandbox.Beam.FEMBeam.FEMBeam import FEMBeam as Beam

if len(sys.argv) > 1:
    scene_map = {'0': 'FEMBeam', '1': 'FEMBeamMouse'}
    arg = '0' if (sys.argv[1] not in scene_map.keys()) and (sys.argv[1] not in scene_map.values()) else sys.argv[1]
    scene = scene_map[arg] if (arg in scene_map.keys()) else arg
    scene_rep = 'Sandbox.FEMBeam.' + scene
    exec(f'from {scene_rep} import {scene} as Beam')


# ENVIRONMENT PARAMETERS
grid_resolution = [25, 5, 5]    # [40, 10, 10]
grid_min = [0., 0., 0.]
grid_max = [100, 15, 15]        # [100, 25, 25]
fixed_box = [0., 0., 0., 0., 25, 25]
p_grid = {'grid_resolution': grid_resolution, 'grid_min': grid_min, 'grid_max': grid_max, 'fixed_box': fixed_box}


def createScene(root_node=None):
    """
    Automatically called when launching a Sofa scene or called from main to create the scene graph.
    :param root_node: Sofa.Core.Node() object.
    :return: root_node
    """
    # Environment config
    env_config = BeamConfig(environment_class=Beam, root_node=root_node, p_grid=p_grid, always_create_data=True,
                            visualizer_class=MeshVisualizer)
    # Manually create and init the environment from the configuration object
    env = env_config.createEnvironment()
    return env.root


# Executed through python interpreter
if __name__ == '__main__':
    # Create scene graph
    root = createScene()
    # Launch the GUI
    Sofa.Gui.GUIManager.Init("main", "qglviewer")
    Sofa.Gui.GUIManager.createGUI(root, __file__)
    Sofa.Gui.GUIManager.SetDimension(1080, 1080)
    Sofa.Gui.GUIManager.MainLoop(root)
    Sofa.Gui.GUIManager.closeGUI()
