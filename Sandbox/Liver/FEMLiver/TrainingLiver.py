"""
TrainingLiver.py
FEM simulated liver with random forces applied on the visible surface.
NN simulated liver which predicts the deformations of the FEM deformed liver.
"""

import copy
import random
import numpy as np
from time import time_ns as timer
import Sofa.Simulation

from DeepPhysX_Sofa.Environment.SofaEnvironment import SofaEnvironment
from Caribou.Topology import Grid3D
from Sandbox.Liver.LiverConfig.utils import extract_visible_nodes, from_sparse_to_regular_grid


# Inherit from SofaEnvironment which allow to implement and create a Sofa scene in the DeepPhysX_Core pipeline
class TrainingLiver(SofaEnvironment):

    def __init__(self, root_node, config, idx_instance=1, visualizer_class=None):
        super(TrainingLiver, self).__init__(root_node, config, idx_instance, visualizer_class)
        # Scene configuration
        self.config = config
        # Keep a track of the actual step number and how many samples diverged during the animation
        self.nb_steps = 0
        self.nb_converged = 0.
        self.converged = False

    def create(self, config):
        """
        Create the Sofa scene graph. Automatically called by SofaEnvironmentConfig.
        :param config: Dataclass of SofaEnvironmentConfig objects, contains the custom parameters of the environment
        :return: None
        """
        # Get the parameters (liver, grid, force)
        p_liver, p_grid, p_force = config.p_liver, config.p_grid, config.p_force

        # UMesh regular grid
        self.regular_grid = Grid3D(anchor_position=p_grid['bbox_anchor'], n=p_grid['nb_cells'],
                                   size=p_grid['bbox_size'])
        print(f"3D grid: {self.regular_grid.number_of_nodes()} nodes, {self.regular_grid.number_of_cells()} cells.")

        # ROOT
        surface_mesh = self.root.addObject('MeshObjLoader', name='surface_mesh', filename=p_liver['mesh_file'],
                                           translation=p_liver['translation'])

        ## FEM NODE ##
        fem = self.root.addChild('fem')
        self.solver = fem.addObject('LegacyStaticODESolver', name='ode_solver', newton_iterations=10, printLog=False,
                                    correction_tolerance_threshold=1e-6, residual_tolerance_threshold=1e-6)
        fem.addObject('ConjugateGradientSolver', name='cg_solver', preconditioning_method='Diagonal',
                      maximum_number_of_iterations=2000, residual_tolerance_threshold=1e-9, printLog=False)
        self.fem_sparse_grid = fem.addObject('SparseGridTopology', name='sparse_grid', src='@../surface_mesh',
                                             n=p_grid['grid_resolution'])
        fem.addObject('BoxROI', name='b_box', box=p_grid['b_box'], drawBoxes=True, drawSize='1.0')
        self.fem_mo = fem.addObject('MechanicalObject', name='fem_mo', src='@sparse_grid', showObject=False)
        fem.addObject('SaintVenantKirchhoffMaterial', name="stvk", young_modulus=5000, poisson_ratio=0.4)
        fem.addObject('HyperelasticForcefield', template="Hexahedron", material="@stvk", topology='@sparse_grid',
                      printLog=True)
        fem.addObject('BoxROI', name='fixed_box', box=p_liver['fixed_box'], drawBoxes=True)
        fem.addObject('FixedConstraint', indices='@fixed_box.indices')
        # surface #
        fem_surface = self.root.fem.addChild('fem_surface')
        self.surface_mo = fem_surface.addObject('MechanicalObject', name='mo_embedded_surface',
                                                src='@../../surface_mesh')
        fem_surface.addObject('TriangleSetTopologyContainer', name='triangles', src='@../../surface_mesh')
        self.sphere = fem_surface.addObject('SphereROI', name='sphere', tags='ROI_SPHERE', radii=0.015,
                                            centers=p_liver['fixed_point'].tolist(), drawSphere=True, drawSize=1)
        self.force_field = fem_surface.addObject('ConstantForceField', name='cff', tags='CFF', indices='0',
                                                 forces=[0., 0., 0.], showArrowSize='0.2', showColor=[0, 0, 1, 1])
        fem_surface.addObject('BarycentricMapping', input='@../fem_mo', output='@./')
        # visible points #
        rgbd = self.root.fem.fem_surface.addChild('rgbd')
        self.visible_nodes = extract_visible_nodes(camera_position=p_liver['camera_position'],
                                                   normals=surface_mesh.normals.value,
                                                   positions=surface_mesh.position.value,
                                                   dot_thresh=0.0, rand_thresh=0.0, distance_from_camera_thresh=1e6)
        rgbd.addObject('MechanicalObject', name='mo_rgbd_pcd', position=self.visible_nodes,
                       showObject=True, showObjectScale=5, showColor=[1, 0, 0, 1])
        rgbd.addObject('BarycentricMapping', input='@../../fem_mo', output='@./')
        # visual #
        visual_node = self.root.fem.addChild('visual')
        self.fem_visu = visual_node.addObject('OglModel', src='@../../surface_mesh', color='green', useVBO=True)
        visual_node.addObject('BarycentricMapping', input='@../fem_mo', output='@./')

        ## NN NODE ##
        network = self.root.addChild('network')
        self.nn_sparse_grid = network.addObject('SparseGridTopology', name='sparse_grid', src='@../surface_mesh',
                                                n=p_grid['grid_resolution'])
        self.nn_mo = network.addObject('MechanicalObject', name='mo', src='@sparse_grid', showObject=False)
        # visual #
        network_visual_node = self.root.network.addChild('visual')
        self.nn_visu = network_visual_node.addObject('OglModel', src='@../../surface_mesh', color='orange', useVBO=True)
        network_visual_node.addObject('BarycentricMapping', input='@../mo', output='@./')

    def onSimulationInitDoneEvent(self, event):
        """
        Called within the Sofa pipeline at the end of the scene graph initialisation.
        :param event: Sofa Event
        :return: None
        """
        # Correspondences between sparse grid and regular grid
        grid_shape = self.config.p_grid['grid_resolution']
        self.nb_nodes_regular_grid = grid_shape[0] * grid_shape[1] * grid_shape[2]
        self.idx_sparse_to_regular, self.idx_regular_to_sparse, \
        self.regular_grid_rest_shape = from_sparse_to_regular_grid(self.nb_nodes_regular_grid, self.fem_sparse_grid,
                                                                   self.fem_mo)
        self.nb_nodes_sparse_grid = len(self.fem_mo.rest_position.value)
        # Get the data sizes
        self.input_size = (self.nb_nodes_regular_grid, 3)
        self.output_size = (self.nb_nodes_regular_grid, 3)
        # Rendering
        self.initVisualizer()

    def initVisualizer(self):
        # Visualizer
        if self.visualizer is not None:
            self.visualizer.addMesh(positions=self.fem_visu.position.value, cells=self.fem_visu.triangles.value, at=0)
            self.visualizer.addMesh(positions=self.nn_visu.position.value, cells=self.nn_visu.triangles.value, at=1)
            self.renderVisualizer()

    def onAnimateBeginEvent(self, event):
        """
        Called within the Sofa pipeline at the beginning of the time step.
        :param event: Sofa Event
        :return: None
        """
        # Reset position
        self.fem_mo.position.value = self.fem_mo.rest_position.value
        # Generate next forces
        f = np.random.uniform(low=-1, high=1, size=(3,))
        f = (f / np.linalg.norm(f)) * self.config.p_force['amplitude_scale'] * np.random.random(1)
        # Pick up a random visible surface point
        current_point = self.visible_nodes[np.random.randint(len(self.visible_nodes))]
        # Set the centers of the ROI sphere to current point
        self.sphere.centers.value = [current_point]
        # Build forces vector
        forces_vector = []
        for i in range(len(self.sphere.indices.array())):
            forces_vector.append(f)
        # Set forces and indices
        self.force_field.indices.value = self.sphere.indices.array()
        self.force_field.forces.value = forces_vector

    def onAnimateEndEvent(self, event):
        """
        Called within the Sofa pipeline at the end of the time step.
        :param event: Sofa Event
        :return: None
        """
        # Count the steps
        self.nb_steps += 1
        # Check whether if the solver diverged or not
        self.converged = self.solver.converged.value
        self.nb_converged += int(self.converged)
        # Render
        self.renderVisualizer()
        # self.computeInput()
        # self.computeOutput()
        # self.applyPrediction(None)

    def computeInput(self):
        """
        Compute the input to be given to the network. Automatically called by EnvironmentManager.
        :return: None
        """
        f = copy.copy(self.force_field.forces.value)
        ind = copy.copy(self.force_field.indices.value)
        positions = self.surface_mo.position.array()[ind]
        F = np.zeros((self.nb_nodes_regular_grid, 3))
        for i in range(len(f)):
            for node in self.regular_grid.node_indices_of(self.regular_grid.cell_index_containing(positions[i])):
                if np.linalg.norm(F) == 0 and node < self.nb_nodes_regular_grid:
                    F[node] = f[i]
        self.input = F

    def computeOutput(self):
        """
        Compute the output to be given to the network. Automatically called by EnvironmentManager.
        :return: None
        """
        actual_positions_on_regular_grid = np.zeros((self.nb_nodes_regular_grid, 3), dtype=np.double)
        actual_positions_on_regular_grid[self.idx_sparse_to_regular] = self.fem_mo.position.array()
        self.output = copy.copy(np.subtract(actual_positions_on_regular_grid,
                                            self.regular_grid_rest_shape))

    def checkSample(self, check_input=True, check_output=True):
        return self.converged

    def applyPrediction(self, prediction):
        """
        Apply the prediction of the network in the Sofa environment. Automatically called by EnvironmentManager.
        :return: None
        """
        # Needed for prediction only
        # U_sparse = self.output[self.idx_sparse_to_regular]
        U = prediction[-1]
        U = np.reshape(U, (self.nb_nodes_regular_grid, 3))
        U_sparse = U[self.idx_sparse_to_regular]
        self.nn_mo.position.value = self.nn_mo.rest_position.array() + U_sparse
