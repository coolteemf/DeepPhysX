import numpy as np
import os

from Sandbox.Liver.Config.utils import compute_grid_resolution


# Liver parameters
filename = os.path.dirname(os.path.abspath(__file__)) + '/liver.obj'
translation = [-0.0134716, 0.021525, -0.427]
fixed_point = np.array([-0.00338, -0.0256, 0.52]) + np.array(translation)
fixed_width = np.array([0.07, 0.05, 0.04])
fixed_box = (fixed_point - fixed_width / 2.).tolist() + (fixed_point + fixed_width / 2.).tolist()
camera_position = np.array([-0.177458, 0.232606, 0.780813])
p_liver = {'mesh_file': filename, 'translation': translation, 'camera_position': camera_position,
           'fixed_box': fixed_box, 'fixed_point': fixed_point}

# Grid parameters
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

# Forces parameters
nb_simultaneous_forces = 20
amplitude_scale = 0.1
inter_distance_thresh = 0.06
p_force = {'nb_simultaneous_forces': nb_simultaneous_forces, 'amplitude_scale': amplitude_scale,
           'inter_distance_thresh': inter_distance_thresh}

# Training parameters
nb_epochs = 100
nb_batch = 30
batch_size = 10
lr = 1e-6
