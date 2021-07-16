"""
liverTrainingUnet.py
Script used to train a Unet network on FEM liver deformations
"""

import os
import torch
import numpy as np
import Sofa
import Sofa.Gui

from Sandbox.Liver.LiverConfig.LiverConfig import LiverConfig
from Sandbox.Liver.NNLiver.NNLiver import NNLiver as Liver
from DeepPhysX_PyTorch.UNet.UnetDataTransformation import UnetDataTransformation
from Sandbox.Liver.LiverConfig.utils import compute_grid_resolution

from DeepPhysX_PyTorch.UNet.UNetConfig import UNetConfig
from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX_Sofa.Runner.SofaRunner import SofaRunner
from DeepPhysX_Core.Visualizer.MeshVisualizer import MeshVisualizer

# ENVIRONMENT PARAMETERS
# Liver
filename = os.path.dirname(os.path.abspath(__file__)) + '/LiverConfig/liver.obj'
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
amplitude_scale = 0.1
inter_distance_thresh = 0.06
p_force = {'nb_simultaneous_forces': nb_simultaneous_forces, 'amplitude_scale': amplitude_scale,
           'inter_distance_thresh': inter_distance_thresh}

# TRAINING PARAMETERS
nb_epoch = 100
nb_batch = 30
batch_size = 10


def createScene(root_node=None):
    # Environment config
    env_config = LiverConfig(environment_class=Liver, root_node=root_node, always_create_data=True,
                             visualizer_class=None, p_liver=p_liver, p_grid=p_grid, p_force=p_force)

    # Network config
    net_config = UNetConfig(network_name="liver_UNet", save_each_epoch=False,
                            loss=torch.nn.MSELoss, lr=1e-6, optimizer=torch.optim.Adam,
                            data_transformation_class=UnetDataTransformation,
                            steps=3, first_layer_channels=128, nb_classes=3,
                            nb_input_channels=3, nb_dims=3, border_mode='same', two_sublayers=True,
                            grid_shape=grid_resolution)
    # Dataset config
    dataset_config = BaseDatasetConfig(partition_size=1, shuffle_dataset=True)
    # Runner
    man_dir = os.path.dirname(os.path.abspath(__file__)) + '/trainings/liver'
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
