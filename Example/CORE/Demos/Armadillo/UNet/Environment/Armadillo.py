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
from time import sleep

# DeepPhysX related imports
from DeepPhysX_Core.Environment.BaseEnvironment import BaseEnvironment
from DeepPhysX_Core.Utils.Visualizer.GridMapping import GridMapping

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
        self.mesh_coarse = None
        self.grid = None
        self.mapping = None
        self.mapping_coarse = None

        # Force fields
        self.forces = []
        self.areas = []
        self.g_areas = []
        self.compute_sample = None

        # Force pattern
        step = 0.5
        self.amplitudes = np.concatenate((np.arange(0, 1, step),
                                          np.arange(1, -1, -step),
                                          np.arange(-1, 0, step)))
        self.idx_amplitude = 0
        self.force_value = None
        self.idx_zone = 0
        self.F = None

        nb_cells = (p_grid.nb_cells[0] + 1) * (p_grid.nb_cells[1] + 1) * (p_grid.nb_cells[2] + 1)
        self.data_size = (nb_cells, 3)

    """
    ENVIRONMENT INITIALIZATION
    Methods will be automatically called in this order to create and initialize Environment.
    """

    def recv_parameters(self, param_dict):

        # Get the model definition parameters
        self.compute_sample = param_dict['compute_sample'] if 'compute_sample' in param_dict else True
        self.amplitudes[0] = 0 if self.compute_sample else 1

    def create(self):

        # Load the meshes
        self.mesh = Mesh(p_model.mesh).scale(p_model.scale)
        self.mesh_coarse = Mesh(p_model.mesh_coarse).scale(p_model.scale)

        # Define regular grid
        grid = [[p_grid.bbox_anchor[0] + i * p_grid.bbox_size[0] / p_grid.nb_cells[0] for i in
                 range(p_grid.nb_cells[0] + 1)],
                [p_grid.bbox_anchor[1] + i * p_grid.bbox_size[1] / p_grid.nb_cells[1] for i in
                 range(p_grid.nb_cells[1] + 1)],
                [p_grid.bbox_anchor[2] + i * p_grid.bbox_size[2] / p_grid.nb_cells[2] for i in
                 range(p_grid.nb_cells[2] + 1)]]
        grid_nodes = [[[[x, y, z] for x in grid[0]] for y in grid[1]] for z in grid[2]]
        grid_nodes = np.array(grid_nodes).reshape(-1, 3)

        cell_corner = lambda x, y, z: z * len(grid[0]) * len(grid[1]) + y * len(grid[0]) + x
        grid_cells = []
        for z in range(p_grid.nb_cells[2]):
            for y in range(p_grid.nb_cells[1]):
                for x in range(p_grid.nb_cells[0]):
                    grid_cells.append([cell_corner(x, y, z),
                                       cell_corner(x + 1, y, z),
                                       cell_corner(x + 1, y + 1, z),
                                       cell_corner(x, y + 1, z),
                                       cell_corner(x, y, z + 1),
                                       cell_corner(x + 1, y, z + 1),
                                       cell_corner(x + 1, y + 1, z + 1),
                                       cell_corner(x, y + 1, z + 1)])
        self.grid = Mesh([grid_nodes, grid_cells])

        # Init mappings
        self.mapping = GridMapping(self.grid, self.mesh)
        self.mapping_coarse = GridMapping(self.grid, self.mesh_coarse)

        # Define force fields
        sphere = lambda x, y: sum([pow(x_i - y_i, 2) for x_i, y_i in zip(x, y)])
        for zone in p_forces.zones:
            # Find the spherical area
            self.areas.append([])
            self.g_areas.append([])
            for i, pts in enumerate(self.mesh_coarse.points()):
                if sphere(pts, p_forces.centers[zone]) <= pow(p_forces.radius[zone], 2):
                    self.areas[-1].append(i)
                    x_cell = np.array(np.array(grid[0]) < pts[0], dtype=int).tolist().index(0) - 1
                    y_cell = np.array(np.array(grid[1]) < pts[1], dtype=int).tolist().index(0) - 1
                    z_cell = np.array(np.array(grid[2]) < pts[2], dtype=int).tolist().index(0) - 1
                    self.g_areas[-1] += [cell_corner(x_cell, y_cell, z_cell),
                                         cell_corner(x_cell, y_cell, z_cell + 1),
                                         cell_corner(x_cell, y_cell + 1, z_cell),
                                         cell_corner(x_cell, y_cell + 1, z_cell + 1),
                                         cell_corner(x_cell + 1, y_cell, z_cell),
                                         cell_corner(x_cell + 1, y_cell, z_cell + 1),
                                         cell_corner(x_cell + 1, y_cell + 1, z_cell),
                                         cell_corner(x_cell + 1, y_cell + 1, z_cell + 1)]
            self.areas[-1] = np.array(self.areas[-1])
            self.g_areas[-1] = np.unique(self.g_areas[-1])
            # Init force value
            self.forces.append(np.zeros((len(self.areas[-1]), 3)))

    def send_visualization(self):

        # Mesh representing detailed Armadillo (object will have id = 0)
        self.factory.add_object(object_type="Mesh",
                                data_dict={"positions": self.mesh.points(),
                                           'cells': self.mesh.cells(),
                                           'wireframe': True,
                                           "c": "orange",
                                           "at": self.instance_id})

        # Arrows representing the force fields (object will have id = 1)
        self.factory.add_object(object_type='Arrows',
                                data_dict={'positions': [0, 0, 0],
                                           'vectors': [0., 0., 0.],
                                           'c': 'green',
                                           'at': self.instance_id})

        # Points representing the grid (object will have id = 2)
        self.factory.add_object(object_type='Points',
                                data_dict={'positions': self.grid.points(),
                                           'r': 3,
                                           'c': 'grey',
                                           'at': self.instance_id})

        # Return the visualization data
        return self.factory.objects_dict

    """
    ENVIRONMENT BEHAVIOR
    Methods will be automatically called by requests from Server or from EnvironmentManager depending on 
    'as_tcp_ip_client' configuration value.
    """

    async def step(self):

        # # Reset forces
        # for i in range(len(self.forces)):
        #     self.forces[i] = zeros((len(self.areas[i]), 3))
        #
        # # Build force vector
        # self.compute_forces()
        #
        # # Compute input
        # areas = []
        # F = zeros(self.data_size)
        # for area, force in zip(self.areas, self.forces):
        #     grid_area = []
        #     for i in area:
        #         position = self.mesh.points()[i]
        #         x_cell = array(array(self.grid[0]) < position[0], dtype=int).tolist().index(0) - 1
        #         y_cell = array(array(self.grid[1]) < position[1], dtype=int).tolist().index(0) - 1
        #         z_cell = array(array(self.grid[2]) < position[2], dtype=int).tolist().index(0) - 1
        #         cell = [self.cell_corner(x_cell, y_cell, z_cell),
        #                 self.cell_corner(x_cell, y_cell, z_cell + 1),
        #                 self.cell_corner(x_cell, y_cell + 1, z_cell),
        #                 self.cell_corner(x_cell, y_cell + 1, z_cell + 1),
        #                 self.cell_corner(x_cell + 1, y_cell, z_cell),
        #                 self.cell_corner(x_cell + 1, y_cell, z_cell + 1),
        #                 self.cell_corner(x_cell + 1, y_cell + 1, z_cell),
        #                 self.cell_corner(x_cell + 1, y_cell + 1, z_cell + 1)]
        #         for node in cell:
        #             if node not in grid_area:
        #                 grid_area.append(node)
        #     F[array(grid_area)] = array(force)[0]
        #
        # # Set training data
        # self.set_training_data(input_array=F.copy(),
        #                        output_array=zeros(self.data_size))

        # Compute a force sample
        if self.compute_sample:
            # Generate a new force
            if self.idx_amplitude == 0:
                self.idx_zone = np.random.randint(0, len(self.forces))
                zone = p_forces.zones[self.idx_zone]
                self.force_value = np.random.uniform(low=-1, high=1, size=(3,)) * p_forces.amplitude[zone]

            # Update current force amplitude
            self.forces[self.idx_zone] = self.force_value * self.amplitudes[self.idx_amplitude]
            self.idx_amplitude = (self.idx_amplitude + 1) % len(self.amplitudes)

            # Create input array
            F = np.zeros(self.data_size)
            F[self.g_areas[self.idx_zone]] = self.forces[self.idx_zone]
            F_s = np.zeros((self.mesh_coarse.N(), 3))
            F_s[self.areas[self.idx_zone]] = self.forces[self.idx_zone]

        # Load a force sample from Dataset
        else:
            # sleep(0.5)
            F = np.zeros(self.data_size) if self.sample_in is None else self.sample_in

        # Set training data
        # self.F = F_s
        self.set_training_data(input_array=F.copy(),
                               output_array=np.zeros(self.data_size))

    def apply_prediction(self, prediction):

        # Reshape prediction
        U = np.reshape(prediction, self.data_size)

        self.update_visual(U)

    def update_visual(self, U):
        # Update surface mesh
        updated_mesh = self.grid.clone().points(self.grid.points().copy() + U)
        mapped_mesh = self.mapping.apply(updated_mesh.points())
        self.factory.update_object_dict(object_id=0,
                                        new_data_dict={'positions': mapped_mesh.points().copy()})

        # Update arrows representing force fields
        # mapped_mesh_coarse = self.mapping_coarse.apply(updated_mesh.points())
        # self.factory.update_object_dict(object_id=1,
        #                                 new_data_dict={'positions': mapped_mesh_coarse.points(),
        #                                                'vectors': 0.25 * self.F / p_model.scale})

        # Update sparse grid positions
        self.factory.update_object_dict(object_id=2,
                                        new_data_dict={'positions': updated_mesh.points().copy()})

        # Send visualization data to update
        self.update_visualisation(visu_dict=self.factory.updated_object_dict)

    def close(self):
        # Shutdown message
        print("Bye!")
