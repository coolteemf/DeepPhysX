"""
parameters
Define the sets of parameters :
    * model parameters
    * forces parameters
"""

import os
import sys
from numpy import array
from collections import namedtuple

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import find_extremities, get_nb_nodes

# Model
mesh = os.path.dirname(os.path.abspath(__file__)) + '/models/armadillo.obj'
sparse_grid = os.path.dirname(os.path.abspath(__file__)) + '/models/sparse_grid.obj'
scale = 1e-3
nb_nodes = get_nb_nodes(sparse_grid)
model = {'mesh': mesh,
         'sparse_grid': sparse_grid,
         'scale': scale,
         'nb_nodes': nb_nodes}
p_model = namedtuple('p_model', model)(**model)

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
          'simultaneous': 2}
p_forces = namedtuple('p_forces', forces)(**forces)
