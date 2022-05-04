"""
Armadillo
Simulation of an Armadillo with deformation predictions from a Fully Connected.
Training data are produced at each time step:
    * input: applied forces on each surface node
    * prediction: resulted displacement of each surface node
"""

# Python related imports
import os
import sys

import numpy as np
from vedo import Mesh
from math import pow
from numpy import array, zeros, reshape, arange
from numpy.random import choice, uniform
from time import sleep

# DeepPhysX related imports
from DeepPhysX_Core.Environment.BaseEnvironment import BaseEnvironment
from DeepPhysX_Core.Utils.Visualizer.barycentric_mapping import BarycentricMapping

# Session related imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from parameters import p_model, p_forces, p_grid


# Create an Environment as a BaseEnvironment child class
class Armadillo(BaseEnvironment):

    def __init__(self,
                 ip_address='localhost',
                 port=10000,
                 instance_id=0,
                 number_of_instances=1,
                 as_tcp_ip_client=True,
                 environment_manager=None):

        BaseEnvironment.__init__(self,
                                 ip_address=ip_address,
                                 port=port,
                                 instance_id=instance_id,
                                 number_of_instances=number_of_instances,
                                 as_tcp_ip_client=as_tcp_ip_client,
                                 environment_manager=environment_manager)

        # Topology
        self.mesh = None
        self.grid = None
        self.cell_corner = None
        self.detailed = True
        self.mapping = None

        # Force fields
        self.forces = []
        self.areas = []
        self.compute_forces = None
        self.current_forces = []

        # Amplitudes pattern
        step = 0.2
        self.amplitudes = arange(0, 1, step).tolist() + arange(1, -1, -step).tolist() + arange(-1, 0, step).tolist()
        self.idx_amplitude = 0
        # Directions pattern
        self.directions = [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]
        self.idx_direction = 0
        # Zone index
        self.idx_zone = 0

        nb_cells = (p_grid.nb_cells[0] + 1) * (p_grid.nb_cells[1] + 1) * (p_grid.nb_cells[2] + 1)
        self.data_size = (nb_cells, 3)

    """
    ENVIRONMENT INITIALIZATION
    Methods will be automatically called in this order to create and initialize Environment.
    """

    def recv_parameters(self, param_dict):

        # Get the model definition parameters
        self.detailed = param_dict['detailed'] if 'detailed' in param_dict else self.detailed
        follow_pattern = param_dict['pattern'] if 'pattern' in param_dict else True
        self.compute_forces = self.compute_pattern_forces if follow_pattern else self.compute_random_forces

    def create(self):

        # Load the coarse mesh and init the barycentric mapping with the detailed mesh
        self.mesh = Mesh(p_model.mesh_coarse).scale(p_model.scale)
        if self.detailed:
            mapped_mesh = Mesh(p_model.mesh).scale(p_model.scale)
            self.mapping = BarycentricMapping(self.mesh, mapped_mesh)

        # Define regular grid
        self.grid = [[p_grid.bbox_anchor[0] + i * p_grid.bbox_size[0] / p_grid.nb_cells[0] for i in
                      range(p_grid.nb_cells[0] + 1)],
                     [p_grid.bbox_anchor[1] + i * p_grid.bbox_size[1] / p_grid.nb_cells[1] for i in
                      range(p_grid.nb_cells[1] + 1)],
                     [p_grid.bbox_anchor[2] + i * p_grid.bbox_size[2] / p_grid.nb_cells[2] for i in
                      range(p_grid.nb_cells[2] + 1)]]
        self.grid_positions = []
        for z in self.grid[2]:
            for y in self.grid[1]:
                for x in self.grid[0]:
                    self.grid_positions.append(array([x, y, z]))
        self.grid_positions = array(self.grid_positions)
        self.cell_corner = lambda x, y, z: z * len(self.grid[0]) * len(self.grid[1]) + y * len(self.grid[0]) + x

        # Define force fields
        sphere = lambda x, y: sum([pow(x_i - y_i, 2) for x_i, y_i in zip(x, y)])
        for zone in p_forces.zones:
            # Find the spherical area
            self.areas.append([])
            for i, pts in enumerate(self.mesh.points()):
                if sphere(pts, p_forces.centers[zone]) <= pow(p_forces.radius[zone], 2):
                    self.areas[-1].append(i)
            # Init force value
            self.forces.append(zeros((len(self.areas[-1]), 3)))

    def send_visualization(self):

        # Mesh representing detailed Armadillo (object will have id = 0)
        mapped_mesh = self.mapping.apply(self.mesh.points().copy()) if self.detailed else self.mesh.clone()
        self.factory.add_object(object_type="Mesh",
                                data_dict={"positions": mapped_mesh.points(),
                                           'cells': mapped_mesh.cells(),
                                           'wireframe': True,
                                           "c": "orange",
                                           "at": self.instance_id})

        # Arrows representing the force fields (object will have id = 1)
        self.factory.add_object(object_type='Arrows',
                                data_dict={'positions': [0, 0, 0],
                                           'vectors': [0., 0., 0.],
                                           'c': 'green',
                                           'at': self.instance_id})

        # Grid
        self.factory.add_object(object_type='Points',
                                data_dict={'positions': self.grid_positions,
                                           'r': 3,
                                           'c': 'grey',
                                           'at': self.instance_id})

        # Prediction
        self.factory.add_object(object_type='Arrows',
                                data_dict={'positions': [0, 0, 0],
                                           'vectors': [0., 0., 0.],
                                           'c': 'red',
                                           'at': self.instance_id})

        # Return the visualization data
        return self.factory.objects_dict

    """
    ENVIRONMENT BEHAVIOR
    Methods will be automatically called by requests from Server or from EnvironmentManager depending on 
    'as_tcp_ip_client' configuration value.
    """

    async def step(self):

        # Reset forces
        for i in range(len(self.forces)):
            self.forces[i] = zeros((len(self.areas[i]), 3))

        # Build force vector
        self.compute_forces()

        # Compute input
        areas = []
        F = zeros(self.data_size)
        for area, force in zip(self.areas, self.forces):
            grid_area = []
            for i in area:
                position = self.mesh.points()[i]
                x_cell = array(array(self.grid[0]) < position[0], dtype=int).tolist().index(0) - 1
                y_cell = array(array(self.grid[1]) < position[1], dtype=int).tolist().index(0) - 1
                z_cell = array(array(self.grid[2]) < position[2], dtype=int).tolist().index(0) - 1
                cell = [self.cell_corner(x_cell, y_cell, z_cell),
                        self.cell_corner(x_cell, y_cell, z_cell + 1),
                        self.cell_corner(x_cell, y_cell + 1, z_cell),
                        self.cell_corner(x_cell, y_cell + 1, z_cell + 1),
                        self.cell_corner(x_cell + 1, y_cell, z_cell),
                        self.cell_corner(x_cell + 1, y_cell, z_cell + 1),
                        self.cell_corner(x_cell + 1, y_cell + 1, z_cell),
                        self.cell_corner(x_cell + 1, y_cell + 1, z_cell + 1)]
                for node in cell:
                    if node not in grid_area:
                        grid_area.append(node)
            F[array(grid_area)] = array(force)[0]

        # Set training data
        self.set_training_data(input_array=F.copy(),
                               output_array=zeros(self.data_size))

    def compute_pattern_forces(self):

        # Get current zone
        zone = p_forces.zones[self.idx_zone]

        # Define next force value
        f = array([0., 0., 0.])
        for direction in self.directions[self.idx_direction]:
            f[direction] = self.amplitudes[self.idx_amplitude] * p_forces.amplitude[zone]
        self.forces[self.idx_zone] = [f.tolist()] * len(self.areas[self.idx_zone])

        # Increment direction index
        if self.idx_amplitude == len(self.amplitudes) - 1 and self.idx_zone == len(self.areas) - 1:
            self.idx_direction = (self.idx_direction + 1) % len(self.directions)
        # Increment zone index
        if self.idx_amplitude == len(self.amplitudes) - 1:
            self.idx_zone = (self.idx_zone + 1) % len(self.areas)
        # Increment amplitude index
        self.idx_amplitude = (self.idx_amplitude + 1) % len(self.amplitudes)

    def compute_random_forces(self):

        # Pick random zone
        zones = choice(len(self.areas), size=p_forces.simultaneous, replace=False)

        # Define next force value
        for i in zones:
            f = uniform(low=-1, high=1, size=(3,))
            f = f * p_forces.amplitude[p_forces.zones[i]]
            self.forces[i] = [f.tolist()] * len(self.areas[i])

        # Time to visualize
        sleep(0.5)

    def apply_prediction(self, prediction):

        # Reshape prediction
        U = reshape(prediction, self.data_size)

        # Update detailed mesh visualization
        new_points = []

        from scipy.interpolate import RegularGridInterpolator
        x = np.linspace(self.grid[0][0], self.grid[0][-1], len(self.grid[0]))
        y = np.linspace(self.grid[1][0], self.grid[1][-1], len(self.grid[1]))
        z = np.linspace(self.grid[2][0], self.grid[2][-1], len(self.grid[2]))
        # X, Y, Z = np.meshgrid(x, y, z, indexing='ij', sparse=True)
        pred = reshape(U, (p_grid.nb_cells[0] + 1, p_grid.nb_cells[1] + 1, p_grid.nb_cells[2] + 1, -1))
        interpolator = RegularGridInterpolator([x, y, z], pred, method='nearest')

        for position in self.mesh.points():
            # x_cell = array(array(self.grid[0]) < position[0], dtype=int).tolist().index(0) - 1
            # y_cell = array(array(self.grid[1]) < position[1], dtype=int).tolist().index(0) - 1
            # z_cell = array(array(self.grid[2]) < position[2], dtype=int).tolist().index(0) - 1
            #
            #
            # # cell = [self.cell_corner(x_cell,     y_cell,     z_cell),
            # #         self.cell_corner(x_cell + 1, y_cell + 1, z_cell + 1),
            # #         self.cell_corner(x_cell,     y_cell,     z_cell + 1),
            # #         self.cell_corner(x_cell + 1, y_cell + 1, z_cell),
            # #         self.cell_corner(x_cell,     y_cell + 1, z_cell),
            # #         self.cell_corner(x_cell + 1, y_cell,     z_cell + 1),
            # #         self.cell_corner(x_cell,     y_cell + 1, z_cell + 1),
            # #         self.cell_corner(x_cell + 1, y_cell,     z_cell)]
            # # interp = vedo.utils.linInterpolate(position,
            # #                                      [self.grid_positions[cell[0]], self.grid_positions[cell[1]]],
            # #                                      [U[cell[0]], U[cell[1]]])
            # D = []
            # for x in (-1, 0, 1, 2):
            #     for y in (-1, 0, 1, 2):
            #         for z in (-1, 0, 1, 2):
            #             D.append(U[self.cell_corner(x_cell + x, y_cell + y, z_cell + z)])
            # U_norms = np.linalg.norm(array(D), axis=1)
            # interp = D[list(U_norms).index(max(U_norms))]

            interp = interpolator(position)

            new_points.append(position + interp[0])

        updated_mesh = self.mesh.clone().points(new_points)
        mapped_mesh = self.mapping.apply(updated_mesh.points()) if self.detailed else updated_mesh
        self.factory.update_object_dict(object_id=0,
                                        new_data_dict={'positions': mapped_mesh.points().copy()})

        # Update arrows representing force fields
        position, vec = [], []
        for area, force in zip(self.areas, self.forces):
            if list(force[0]) != [0., 0., 0.]:
                position += list(updated_mesh.points()[array(area)])
                vec += list(0.25 * array(force) / p_model.scale)
        if len(position) == 0 or len(vec) == 0:
            position, vec = [0., 0., 0.], [0., 0., 0.]
        self.factory.update_object_dict(object_id=1,
                                        new_data_dict={'positions': position,
                                                       'vectors': vec})

        # Update arrows representing displacement field
        self.factory.update_object_dict(object_id=3,
                                        new_data_dict={'positions': self.grid_positions,
                                                       'vectors': U})

        # Send visualization data to update
        self.update_visualisation(visu_dict=self.factory.updated_object_dict)

    def close(self):

        # Shutdown message
        print("Bye!")
