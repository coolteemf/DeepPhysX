"""
runSofa.py
Run this script to check if the Sofa scene is working well. Outside of DeepPhysX_Core pipeline, only run scenes in this
repository.
The default scene is FEMLiver.py, but other scene can be launched using 'python3 runSofa.py x' with x being either the
name of the scene either it's corresponding number in the following scene map dictionary.
"""

import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import Sofa.Gui

from DeepPhysX_Core.Visualizer.MeshVisualizer import MeshVisualizer

from Sandbox.Liver.LiverConfig.LiverConfig import LiverConfig
from Sandbox.Liver.FEMLiver.FEMLiver import FEMLiver as Liver
from Sandbox.Liver.LiverConfig.utils import compute_grid_resolution

if len(sys.argv) > 1:
    scene_map = {'0': 'FEMLiver'}
    arg = '0' if (sys.argv[1] not in scene_map.keys()) and (sys.argv[1] not in scene_map.values()) else sys.argv[1]
    scene = scene_map[arg] if (arg in scene_map.keys()) else arg
    scene_rep = 'Sandbox.FEMLiver.' + scene
    exec(f'from {scene_rep} import {scene} as Liver')

# ENVIRONMENT PARAMETERS
# Liver
filename = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/LiverConfig/liver.obj'
translation = [-0.0134716, 0.021525, -0.427]
fixed_point = np.array([-0.00338, -0.0256, 0.52]) + np.array(translation)
fixed_width = np.array([0.07, 0.05, 0.04])
fixed_box = (fixed_point - fixed_width / 2.).tolist() + (fixed_point + fixed_width / 2.).tolist()
camera_position = np.array([-0.177458, 0.232606, 0.780813])
p_liver = {'mesh_file': filename, 'translation': translation, 'camera_position': camera_position,
           'fixed_box': fixed_box, 'fixed_point': fixed_point}
# Grid variables
margins = np.array([0.02, 0.02, 0.02])
min_bbox = np.array([-0.130815, -0.107192, 0.00732511]) - margins
max_bbox = np.array([0.0544588, 0.0967464, 0.15144]) + margins
bbox_size = max_bbox - min_bbox
b_box = min_bbox.tolist() + max_bbox.tolist()
cell_size = 0.07
grid_resolution = compute_grid_resolution(max_bbox, min_bbox, cell_size)
print("Grid resolution is {}".format(grid_resolution))
nb_cells_x = grid_resolution[0] - 1
nb_cells_y = grid_resolution[1] - 1
nb_cells_z = grid_resolution[2] - 1
p_grid = {'b_box': b_box, 'bbox_anchor': min_bbox.tolist(), 'bbox_size': bbox_size,
          'nb_cells': [nb_cells_x, nb_cells_y, nb_cells_z], 'grid_resolution': grid_resolution}
# Forces variables
nb_simultaneous_forces = 20
amplitude_scale = 0.05
inter_distance_thresh = 0.06
p_force = {'nb_simultaneous_forces': nb_simultaneous_forces, 'amplitude_scale': amplitude_scale,
           'inter_distance_thresh': inter_distance_thresh}


def createScene(root_node=None):
    """
    Automatically called when launching a Sofa scene or called from main to create the scene graph.
    :param root_node: Sofa.Core.Node() object.
    :return: root_node
    """
    # Environment config
    env_config = LiverConfig(environment_class=Liver, root_node=root_node, always_create_data=True,
                             p_liver=p_liver, p_grid=p_grid, p_force=p_force, visualizer_class=None)
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
