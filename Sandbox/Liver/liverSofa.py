"""
liverSofa.py
Run this script outside of any DPX pipeline, use Sofa to check the scene behaviour.
The default scene is FEMLiver.py, but other scene can be added in the 'scene_map' dictionary.
Run with:
    'python3 liverSofa.py x'
    with x being either a scene name either it's corresponding number in the 'scene_map' dictionary.
"""

import os
import sys
sys.path.append(os.getcwd())
import Sofa.Gui

from Sandbox.Liver.Config.LiverConfig import LiverConfig
from Sandbox.Liver.Scene.BothLiver import BothLiver as Liver
from Sandbox.Liver.Config.parameters import *

from DeepPhysX_Core.Visualizer.MeshVisualizer import MeshVisualizer


# Configure script
if len(sys.argv) > 1:
    scene_map = {'0': 'BothLiver', '1': 'FEMLiver'}
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
                             p_liver=p_liver, p_grid=p_grid, p_force=p_force, visualizer_class=MeshVisualizer)
    # Manually create and init the environment from the configuration object
    env = env_config.createEnvironment()
    # Todo: remove once visualizer initialisation will be automatically added in SofaEnvironmentConfig
    env.visualizer = env_config.visualizer_class()
    env.initVisualizer()
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
