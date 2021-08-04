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

from DeepPhysX_Core.Visualizer.MeshVisualizer import MeshVisualizer
from DeepPhysX_Sofa.Environment.SofaEnvironmentConfig import SofaEnvironmentConfig
from DeepPhysX_Core.Manager.EnvironmentManager import EnvironmentManager
# from Sandbox.Liver.Scene.BothLiverF import BothLiverF as Liver
from Sandbox.Liver.Scene.BothLiverD import BothLiverD as Liver


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
    env_config = SofaEnvironmentConfig(environment_class=Liver, root_node=root_node, always_create_data=True)
    # Manually create and init the environment from the configuration object
    # env = env_config.createEnvironment()
    # env_config.initSofaSimulation()
    # env.initVisualizer()
    # return env.root
    env_manager = EnvironmentManager(environment_config=env_config)
    return env_manager.environment.root


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
