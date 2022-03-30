"""
Parameters
Define the sets of parameters :
    * model parameters
    * grid parameters
    * forces parameters
"""

import os
import sys
from numpy import array
from collections import namedtuple

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from DeepPhysX.Example.SOFA.Demos.Armadillo.Environment.utils import define_bbox, compute_grid_resolution, \
    find_fixed_box, find_extremities, get_nb_nodes, \
    get_object_max_size

# Model
mesh = os.path.dirname(os.path.abspath(__file__)) + '/models/armadillo.obj'
coarse_mesh = os.path.dirname(os.path.abspath(__file__)) + '/models/armadillo_coarse.obj'
scale = 1e-3
scale3d = 3 * [scale]
size = get_object_max_size(mesh, scale)
fixed_box = find_fixed_box(mesh, scale)
nb_nodes = get_nb_nodes(coarse_mesh)
model = {'mesh': mesh,
         'mesh_coarse': coarse_mesh,
         'scale': scale,
         'scale3d': scale3d,
         'size': size,
         'fixed_box': fixed_box,
         'nb_nodes': nb_nodes}
p_model = namedtuple('p_model', model)(**model)

# Grid
margin_scale = 0.1
cell_size = 0.06
min_bbox, max_bbox, b_box = define_bbox(mesh, margin_scale, scale)
grid_resolution = compute_grid_resolution(max_bbox, min_bbox, cell_size)
grid = {'b_box': b_box,
        'grid_resolution': grid_resolution}
p_grid = namedtuple('p_grid', grid)(**grid)

# Forces
zones = ['tail', 'r_hand', 'l_hand', 'r_ear', 'l_ear', 'muzzle']
centers, radius, amplitude = {}, {}, {}
for zone, c, rad, amp in zip(zones, find_extremities(mesh, scale), [2.5, 2.5, 2.5, 2., 2., 1.5],
                             array([15, 2.5, 2.5, 7.5, 7.5, 7.5]) * scale):
    centers[zone] = c
    radius[zone] = scale * rad
    amplitude[zone] = scale * amp
forces = {'zones': zones,
          'centers': centers,
          'radius': radius,
          'amplitude': amplitude,
          'simultaneous': 1}
p_forces = namedtuple('p_forces', forces)(**forces)
