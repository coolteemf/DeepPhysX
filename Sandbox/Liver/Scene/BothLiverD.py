"""
BothLiver.py
FEM simulated liver with random forces applied on the visible surface.
NN simulated liver which predicts the deformations of the FEM deformed liver.
"""

import os
import copy
import numpy as np
import random
import Sofa.Simulation
import SofaRuntime

from DeepPhysX_Sofa.Environment.SofaEnvironment import SofaEnvironment
from DeepPhysX_Core.Visualizer.MeshVisualizer import MeshVisualizer
from Caribou.Topology import Grid3D
from Sandbox.Liver.Config.utils import extract_visible_nodes, from_sparse_to_regular_grid
from Sandbox.Liver.Config.parameters import p_liver, p_grid, p_force


# Inherit from SofaEnvironment which allow to implement and create a Sofa scene in the DeepPhysX_Core pipeline
class BothLiverD(SofaEnvironment):

    def __init__(self, root_node, config, idx_instance=1):
        super(BothLiverD, self).__init__(root_node, config, idx_instance)
        # Keep a track of the actual step number and how many samples diverged during the animation
        self.nb_steps = 0
        self.nb_converged = 0.
        self.converged = True
        self.is_created = {'fem': False, 'nn': False}

    def create(self):
        """
        Create the Sofa scene graph. Automatically called by SofaEnvironmentConfig.

        :return: None
        """

        # UMesh regular grid
        self.regular_grid = Grid3D(anchor_position=p_grid['bbox_anchor'], n=p_grid['nb_cells'],
                                   size=p_grid['bbox_size'])
        print(f"3D grid: {self.regular_grid.number_of_nodes()} nodes, {self.regular_grid.number_of_cells()} cells.")

        # Required plugins
        SofaRuntime.PluginRepository.addFirstPath(os.environ['CARIBOU_INSTALL'])
        required_plugins = ['SofaComponentAll', 'SofaLoader', 'SofaCaribou', 'SofaBaseTopology', 'SofaGeneralEngine',
                            'SofaEngine', 'SofaOpenglVisual', 'SofaBoundaryCondition']
        self.root.addObject('RequiredPlugin', pluginName=required_plugins)

        # Root
        self.surface_mesh = self.root.addObject('MeshObjLoader', name='surface_mesh', filename=p_liver['mesh_file'],
                                                translation=p_liver['translation'])
        self.visible_nodes = extract_visible_nodes(camera_position=p_liver['camera_position'],
                                                   normals=self.surface_mesh.normals.value,
                                                   positions=self.surface_mesh.position.value,
                                                   dot_thresh=0.0, rand_thresh=0.0, distance_from_camera_thresh=1e6)
        # Root children
        self.addModels(p_liver, p_grid)

    def addModels(self, p_liver, p_grid):
        self.createFEM(p_liver, p_grid)
        self.createNN(p_liver, p_grid)
        self.is_created['fem'], self.is_created['nn'] = True, True

    def createFEM(self, p_liver, p_grid):
        """
        FEM node of the liver scene

        :param p_liver:
        :param p_grid:
        :return:
        """
        fem = self.root.addChild('fem')
        # Solvers
        self.solver = fem.addObject('LegacyStaticODESolver', name='ode_solver', newton_iterations=10, printLog=False,
                                    correction_tolerance_threshold=1e-6, residual_tolerance_threshold=1e-6)
        fem.addObject('ConjugateGradientSolver', name='cg_solver', preconditioning_method='Diagonal',
                      maximum_number_of_iterations=2000, residual_tolerance_threshold=1e-9, printLog=False)
        # Topology
        self.fem_sparse_grid = fem.addObject('SparseGridTopology', name='sparse_grid', src='@../surface_mesh',
                                             n=p_grid['grid_resolution'])
        self.fem_mo = fem.addObject('MechanicalObject', name='fem_mo', src='@sparse_grid', showObject=False)
        fem.addObject('BoxROI', name='b_box', box=p_grid['b_box'], drawBoxes=True, drawSize='1.0')
        # Material
        fem.addObject('SaintVenantKirchhoffMaterial', name="stvk", young_modulus=5000, poisson_ratio=0.4)
        fem.addObject('HyperelasticForcefield', template="Hexahedron", material="@stvk", topology='@sparse_grid',
                      printLog=True)
        # Constraint
        fem.addObject('BoxROI', name='fixed_box', box=p_liver['fixed_box'], drawBoxes=True)
        fem.addObject('FixedConstraint', indices='@fixed_box.indices')
        # Surface
        fem_surface = self.root.fem.addChild('fem_surface')
        self.fem_surface_mo = fem_surface.addObject('MechanicalObject', name='mo_embedded_surface',
                                                    src='@../../surface_mesh')
        self.fem_surface = fem_surface.addObject('TriangleSetTopologyContainer', name='triangles',
                                                 src='@../../surface_mesh')
        self.fem_sphere = fem_surface.addObject('SphereROI', name='sphere', tags='ROI_SPHERE', radii=0.015,
                                                centers=p_liver['fixed_point'].tolist(), drawSphere=True, drawSize=1)
        self.fem_force_field = fem_surface.addObject('ConstantForceField', name='cff', tags='CFF', indices='0',
                                                     forces=[0., 0., 0.], showArrowSize='0.2', showColor=[0, 0, 1, 1])
        fem_surface.addObject('BarycentricMapping', input='@../fem_mo', output='@./')
        # Visible points
        rgbd = self.root.fem.fem_surface.addChild('rgbd')
        self.points_cloud = rgbd.addObject('MechanicalObject', name='mo_rgbd_pcd', position=self.visible_nodes,
                                           showObject=True, showObjectScale=5, showColor=[1, 0, 0, 1])
        print("Points cloud has ", len(self.points_cloud.position.array()))
        rgbd.addObject('BarycentricMapping', input='@../../fem_mo', output='@./')
        # Visual
        visual_node = self.root.fem.addChild('visual')
        self.fem_visu = visual_node.addObject('OglModel', src='@../../surface_mesh', color='green')
        visual_node.addObject('BarycentricMapping', input='@../fem_mo', output='@./')

    def createNN(self, p_liver, p_grid):
        """
        Neural Network part of the liver

        :param p_liver:
        :param p_grid:
        :return:
        """
        network = self.root.addChild('network')
        # Fake solver
        # network.addObject('LegacyStaticODESolver', name='StaticSolver', newton_iterations=0, printLog=False)
        # Topology
        self.nn_sparse_grid = network.addObject('SparseGridTopology', name='sparse_grid', src='@../surface_mesh',
                                                n=p_grid['grid_resolution'])
        self.nn_mo = network.addObject('MechanicalObject', name='mo', src='@sparse_grid', showObject=False)
        network.addObject('BoxROI', name='b_box', box=p_grid['b_box'], drawBoxes=True, drawSize='1.0')
        # Surface
        network_embedded_surface = self.root.network.addChild('network_embedded_surface')
        self.nn_surface_mo = network_embedded_surface.addObject('MechanicalObject', name='mo_embedded_surface',
                                                                src='@../../surface_mesh')
        network_embedded_surface.addObject('TriangleSetTopologyContainer', name='triangle_topo',
                                           src='@../../surface_mesh')
        self.nn_sphere = network_embedded_surface.addObject('SphereROI', name='sphere', tags='ROI_SPHERE',
                                                            centers=p_liver['fixed_point'].tolist(), radii=0.015,
                                                            drawSphere=True, drawSize=1)
        self.nn_force_field = network_embedded_surface.addObject('ConstantForceField', name='cff', tags='CFF',
                                                                 indices='0', forces=[0., 0., 0.],
                                                                 showArrowSize='0.2', showColor=[1, 0, 1, 1])
        network_embedded_surface.addObject('BarycentricMapping', input='@../mo', output='@./')
        # Visual
        network_visual_node = self.root.network.addChild('visual')
        self.nn_visu = network_visual_node.addObject('OglModel', src='@../../surface_mesh', color='orange')
        network_visual_node.addObject('BarycentricMapping', input='@../mo', output='@./')

    def onSimulationInitDoneEvent(self, event):
        """
        Called within the Sofa pipeline at the end of the scene graph initialisation.

        :param event: Sofa Event
        :return: None
        """
        print("I am stupid and I do an init each time I'm told to do so")
        # Correspondences between sparse grid and regular grid
        grid_shape = p_grid['grid_resolution']
        mo = self.fem_mo if self.is_created['fem'] else self.nn_mo
        sparse_grid = self.fem_sparse_grid if self.is_created['fem'] else self.nn_sparse_grid
        self.nb_nodes_regular_grid = grid_shape[0] * grid_shape[1] * grid_shape[2]
        self.idx_sparse_to_regular, self.idx_regular_to_sparse, \
        self.regular_grid_rest_shape = from_sparse_to_regular_grid(self.nb_nodes_regular_grid, sparse_grid, mo)
        self.nb_nodes_sparse_grid = len(mo.rest_position.value)
        # Get the data sizes
        self.input_size = (self.nb_nodes_regular_grid, 1)
        self.output_size = (self.nb_nodes_regular_grid, 3)
        # Rendering
        self.initVisualizer()

    def initVisualizer(self):
        # Visualizer
        if self.getDataManager() is not None:
            self.visualizer = self.getDataManager().visualizer_manager.visualizer
        else:
            self.visualizer = MeshVisualizer()
        if self.is_created['fem']:
            self.visualizer.addObject(positions=self.fem_surface_mo.position.value,
                                      cells=self.fem_surface.triangles.value)
            # self.visualizer.addObject(positions=self.fem_surface_mo.position.value, at=1)
        if self.is_created['nn']:
            self.visualizer.addObject(positions=self.nn_visu.position.value, cells=self.nn_visu.triangles.value)
        if self.getDataManager() is None:
            self.visualizer.render()

    def onAnimateBeginEvent(self, event):
        """
        Called within the Sofa pipeline at the beginning of the time step.

        :param event: Sofa Event
        :return: None
        """
        # Reset position
        if self.is_created['fem']:
            self.fem_mo.position.value = self.fem_mo.rest_position.value
        # Generate next forces
        f = np.random.uniform(low=-1, high=1, size=(3,))
        f = (f / np.linalg.norm(f)) * p_force['amplitude_scale'] * np.random.random(1)
        # Pick up a random visible surface point
        current_point = self.visible_nodes[np.random.randint(len(self.visible_nodes))]
        # Fem or nn
        sphere = self.fem_sphere if self.is_created['fem'] else self.nn_sphere
        force_field = self.fem_force_field if self.is_created['fem'] else self.nn_force_field
        # Set the centers of the ROI sphere to current point
        sphere.centers.value = [current_point]
        # Build forces vector
        forces_vector = []
        for i in range(len(sphere.indices.array())):
            forces_vector.append(f)
        # Set forces and indices
        force_field.indices.value = sphere.indices.array()
        force_field.forces.value = forces_vector

    def onAnimateEndEvent(self, event):
        """
        Called within the Sofa pipeline at the end of the time step.

        :param event: Sofa Event
        :return: None
        """
        # Count the steps
        self.nb_steps += 1
        # Check whether if the solver diverged or not
        if self.is_created['fem']:
            self.converged = self.solver.converged.value
            if not self.converged:
                Sofa.Simulation.reset(self.root)
            self.nb_converged += int(self.converged)
        # Render
        self.visualizer.render()

    def computeInput(self):
        """
        Compute the input to be given to the network. Automatically called by EnvironmentManager.

        :return: None
        """
        # SELECTION
        # Select a random amount of points on the point cloud
        indices = [random.randint(0, len(self.points_cloud.position.array())-1)
                   for _ in range(random.randint(50, 800))]
        indices = np.unique(np.array(indices))
        actual_positions_of_point_cloud = self.points_cloud.position.array()[indices]
        # ENCODING
        # Init distance field to zero
        DF_grid = np.zeros((self.nb_nodes_regular_grid, 1))
        # Get the list of nodes composing a cell containing a point from the RGBD point cloud
        # For each node of the cell, compute the distance to the considered point
        for p in actual_positions_of_point_cloud:
            cell = self.regular_grid.cell_index_containing(p)
            for node in self.regular_grid.node_indices_of(cell):
                if node < self.nb_nodes_regular_grid and DF_grid[node][0] == 0:
                    DF_grid[node] = np.linalg.norm(self.regular_grid.node(node) - p)
        self.input = DF_grid

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
        U = prediction[-1]
        U = np.reshape(U, (self.nb_nodes_regular_grid, 3))
        U_sparse = U[self.idx_sparse_to_regular]
        self.nn_mo.position.value = self.nn_mo.rest_position.array() + U_sparse

    def close(self):
        pass
