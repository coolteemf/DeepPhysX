import os
import numpy as np

from Application.LiverRegistration.SofaScene.utils import compute_grid_resolution, find_boundaries, define_bbox

# Liver parameters
filename = os.path.dirname(os.path.abspath(__file__)) + '/liver.vtk'
boundary_files = [os.path.dirname(os.path.abspath(__file__)) + '/venacava.vtk']
translation = np.array([0., 0., 0.])
vedo_translation = [0., 0.25, 0.]
min_corner, max_corner, fixed_box = find_boundaries(filename, boundary_files, translation)
fixed_point = min_corner + 0.5 * (max_corner - min_corner)
# fixed_box = [10e-4 * x for x in fixed_box]
p_liver = {'mesh_file': filename,
           'translation': translation.tolist(),
           'vedo_translation': vedo_translation,
           'fixed_box': fixed_box,
           'fixed_point': fixed_point}

# Grid parameters
margin_scale = 0.1
min_bbox, max_bbox, b_box = define_bbox(filename, margin_scale)
bbox_size = max_bbox - min_bbox
cell_size = 0.07
grid_resolution = compute_grid_resolution(max_bbox, min_bbox, cell_size)
nb_cells_x = grid_resolution[0] - 1
nb_cells_y = grid_resolution[1] - 1
nb_cells_z = grid_resolution[2] - 1
p_grid = {'b_box': b_box, 'bbox_anchor': min_bbox.tolist(), 'bbox_size': bbox_size,
          'nb_cells': [nb_cells_x, nb_cells_y, nb_cells_z], 'grid_resolution': grid_resolution}

# Camera parameters
camera_position = np.array([fixed_point[0], fixed_point[1], fixed_point[2] - 1.5 * bbox_size[2]])
camera_thresholds = [0.75, 300]
max_occlusions = 3
min_max_occlusion_proportions = [0.05, 0.3]
noise_level = 7.5e-4
p_camera = {'camera_position': camera_position, 'camera_thresholds': camera_thresholds,
            'max_occlusions': max_occlusions, 'min_max_occlusion_proportions': min_max_occlusion_proportions,
            'noise_level': noise_level}

# Forces parameters
nb_simultaneous_forces = 20
amplitude_scale = 7500
inter_distance_thresh = 6
p_force = {'nb_simultaneous_forces': nb_simultaneous_forces, 'amplitude_scale': amplitude_scale,
           'inter_distance_thresh': inter_distance_thresh}

# Training parameters
nb_epochs = 50
nb_batch = 400
batch_size = 10
lr = 1e-5
