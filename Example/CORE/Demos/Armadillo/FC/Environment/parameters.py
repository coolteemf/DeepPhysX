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
from utils import get_nb_nodes, find_extremities, get_object_max_size

# Model
mesh = os.path.dirname(os.path.abspath(__file__)) + '/models/armadillo.obj'
coarse_mesh = os.path.dirname(os.path.abspath(__file__)) + '/models/armadillo_coarse.obj'
scale = 1e-3
size = get_object_max_size(mesh, scale)
nb_nodes = get_nb_nodes(coarse_mesh)
model = {'mesh': mesh,
         'mesh_coarse': coarse_mesh,
         'scale': scale,
         'size': size,
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
