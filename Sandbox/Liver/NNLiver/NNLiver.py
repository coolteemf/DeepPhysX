"""
FEMLiver.py
FEM simulated liver with random forces applied on the visible surface.
Can be launched as a Sofa scene using the 'runSofa.py' script in this repository.
Also used to train neural network in DeepPhysX_Core pipeline with the '../liverTrainingUNet.py' script.
"""

import copy
import numpy as np

from Sandbox.Liver.TrainingLiver import TrainingLiver
from Caribou.Topology import Grid3D
from Sandbox.Liver.LiverConfig.utils import extract_visible_nodes, from_sparse_to_regular_grid


# Inherit from SofaEnvironment which allow to implement and create a Sofa scene in the DeepPhysX_Core pipeline
class NNLiver(TrainingLiver):

    def __init__(self, root_node, config, idx_instance=1, visualizer_class=None):
        super(NNLiver, self).__init__(root_node, config, idx_instance, visualizer_class)

    def create(self, config):
        """
        Create the Sofa scene graph. Automatically called by SofaEnvironmentConfig.
        :param config: Dataclass of SofaEnvironmentConfig objects, contains the custom parameters of the environment
        :return: None
        """
        # Get the parameters (liver, grid, force)
        p_liver, p_grid = config.p_liver, config.p_grid

        # UMesh regular grid
        self.regular_grid = Grid3D(anchor_position=p_grid['bbox_anchor'], n=p_grid['nb_cells'],
                                   size=p_grid['bbox_size'])
        print(f"3D grid: {self.regular_grid.number_of_nodes()} nodes, {self.regular_grid.number_of_cells()} cells.")

        # Root
        self.surface_mesh = self.root.addObject('MeshObjLoader', name='surface_mesh', filename=p_liver['mesh_file'],
                                                translation=p_liver['translation'])
        self.visible_nodes = extract_visible_nodes(camera_position=p_liver['camera_position'],
                                                   normals=self.surface_mesh.normals.value,
                                                   positions=self.surface_mesh.position.value,
                                                   dot_thresh=0.0, rand_thresh=0.0, distance_from_camera_thresh=1e6)
        # Root children
        self.createNN(p_liver, p_grid)

    def initVisualizer(self):
        # Visualizer
        if self.visualizer is not None:
            self.visualizer.addMesh(positions=self.nn_visu.position.value, cells=self.nn_visu.triangles.value, at=1)
            self.renderVisualizer()

    def onAnimateBeginEvent(self, event):
        """
        Called within the Sofa pipeline at the beginning of the time step.
        :param event: Sofa Event
        :return: None
        """
        # Generate next forces
        p_force = self.config.p_force
        f = np.random.uniform(low=-1, high=1, size=(3,))
        f = (f / np.linalg.norm(f)) * p_force['amplitude_scale'] * np.random.random(1)

        # Pick up a random visible surface point and apply translation
        current_point = self.visible_nodes[np.random.randint(len(self.visible_nodes))]

        # Set the centers of the ROI sphere to current point
        self.nn_sphere.centers.value = [current_point]

        # Build forces vector
        forces_vector = []
        for i in range(len(self.sphere.indices.array())):
            forces_vector.append(f)

        # Set forces and indices
        self.nn_force_field.indices.value = self.nn_sphere.indices.array()
        self.nn_force_field.forces.value = forces_vector

    def onAnimateEndEvent(self, event):
        """
        Called within the Sofa pipeline at the end of the time step.
        :param event: Sofa Event
        :return: None
        """
        # Count the steps
        self.nb_steps += 1
        # Render
        self.renderVisualizer()

    def computeInput(self):
        """
        Compute the input to be given to the network. Automatically called by EnvironmentManager.
        :return: None
        """
        f = copy.copy(self.nn_force_field.forces.value)
        ind = copy.copy(self.nn_force_field.indices.value)
        positions = self.nn_surface_mo.position.array()[ind]
        F = np.zeros((self.nb_nodes_regular_grid, 3))
        for i in range(len(f)):
            for node in self.grid.node_indices_of(self.grid.cell_index_containing(positions[i])):
                if np.linalg.norm(F[node]) == 0 and node < self.nb_nodes_regular_grid:
                    F[node] = f[i]
        self.input = F

    def computeOutput(self):
        """
        Compute the output to be given to the network. Automatically called by EnvironmentManager.
        :return: None
        """
        self.output = np.zeros((self.nb_nodes_regular_grid, 3), dtype=np.double)

    def applyPrediction(self, prediction):
        """
        Apply the prediction of the network in the Sofa environment. Automatically called by EnvironmentManager.
        :return: None
        """
        # Add the displacement to the initial position
        U = prediction[0]

        # Mapping between regular and sparse grids
        U = np.reshape(U, (self.nb_nodes_regular_grid, 3))
        U_sparse = U[self.idx_sparse_to_regular]
        self.nn_mo.position.value = self.nn_mo.rest_position.array() + U_sparse

        # Render
        self.renderVisualizer()

        # Loss
        # U_target = copy.copy(np.subtract(self.MO.position.array(), self.MO.rest_position.array()))
        # print(np.linalg.norm(np.array(U_sparse)))
        # print(np.linalg.norm(np.array(U_target)))
        # import torch
        # criterion = torch.nn.MSELoss()
        # mse = criterion(torch.from_numpy(U_sparse), torch.from_numpy(U_target))
        # print("Effective loss =", mse.item())
