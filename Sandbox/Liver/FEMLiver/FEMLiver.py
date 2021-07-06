"""
FEMLiver.py
FEM simulated liver with random forces applied on the visible surface.
Can be launched as a Sofa scene using the 'runSofa.py' script in this repository.
Also used to train neural network in DeepPhysX_Core pipeline with the '../liverTrainingUNet.py' script.
"""

import copy
import random
import numpy as np
from time import time_ns as timer

from DeepPhysX_Sofa.Environment.SofaEnvironment import SofaEnvironment
from Caribou.Topology import Grid3D
from Sandbox.Liver.LiverConfig.utils import extract_visible_nodes, from_sparse_to_regular_grid


# Inherit from SofaEnvironment which allow to implement and create a Sofa scene in the DeepPhysX_Core pipeline
class FEMLiver(SofaEnvironment):

    def __init__(self, root_node, config, idx_instance=1, visualizer_class=None):
        super(FEMLiver, self).__init__(root_node, config, idx_instance, visualizer_class)
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

        # /root
        surface_mesh_loader = self.root.addObject('MeshObjLoader', name='surface_mesh', filename=p_liver['mesh_file'],
                                                  translation=p_liver['translation'])
        self.visible_surface_nodes = extract_visible_nodes(camera_position=p_liver['camera_position'],
                                                           normals=surface_mesh_loader.normals.value,
                                                           positions=surface_mesh_loader.position.value,
                                                           dot_thresh=0.0, rand_thresh=0.0,
                                                           distance_from_camera_thresh=1e6)
        self.grid = Grid3D(anchor_position=p_grid['bbox_anchor'], n=p_grid['nb_cells'], size=p_grid['bbox_size'])
        print(f"3D grid: {self.grid.number_of_nodes()} nodes, {self.grid.number_of_cells()} cells.")

        # /root/mechanical
        mechanical = self.root.addChild('mechanical')
        self.solver = mechanical.addObject('LegacyStaticODESolver', name='ode_solver', newton_iterations=10,
                                           correction_tolerance_threshold=1e-6,
                                           residual_tolerance_threshold=1e-6, printLog=False)
        mechanical.addObject('ConjugateGradientSolver', name='cg_solver', preconditioning_method='Diagonal',
                             maximum_number_of_iterations=2000, residual_tolerance_threshold=1e-9,
                             printLog=False)
        self.sparse_grid = mechanical.addObject('SparseGridTopology', name='sparse_grid', src='@../surface_mesh',
                                                n=p_grid['grid_resolution'])
        mechanical.addObject('BoxROI', name='b_box', box=p_grid['b_box'], drawBoxes=True, drawSize='1.0')
        self.behaviour_state = mechanical.addObject('MechanicalObject', name='mo', src='@sparse_grid', showObject=False)
        mechanical.addObject('SaintVenantKirchhoffMaterial', name="stvk", young_modulus=5000, poisson_ratio=0.4)
        mechanical.addObject('HyperelasticForcefield', template="Hexahedron", material="@stvk", topology='@sparse_grid',
                             printLog=True)
        mechanical.addObject('BoxROI', name='fixed_box', box=p_liver['fixed_box'], drawBoxes=True)
        mechanical.addObject('FixedConstraint', indices='@fixed_box.indices')

        # /root/mechanical/embedded_surface
        embedded_surface = self.root.mechanical.addChild('embedded_surface')
        embedded_surface.addObject('MechanicalObject', name='mo_embedded_surface', src='@../../surface_mesh')
        embedded_surface.addObject('TriangleSetTopologyContainer', name='triangle_topo', src='@../../surface_mesh')
        nb_forces = p_force['nb_simultaneous_forces']
        self.spheres = [embedded_surface.addObject('SphereROI', name='sphere' + str(i), tags='ROI_SPHERE' + str(i),
                                                   centers=p_liver['fixed_point'].tolist(), radii=0.015,
                                                   drawSphere=True, drawSize=1)
                        for i in range(nb_forces)]
        self.force_fields = [embedded_surface.addObject('ConstantForceField', name='cff' + str(i), tags='CFF' + str(i),
                                                        indices='0', forces=[0., 0., 0.],
                                                        showArrowSize='0.2', showColor=[1, 0, 1, 1])
                             for i in range(nb_forces)]
        embedded_surface.addObject('BarycentricMapping', input='@../mo', output='@./')

        # /root/mechanical/embedded_surface/visible_points
        visible_points = self.root.mechanical.embedded_surface.addChild('visible_points')
        self.learn_on_mo = visible_points.addObject('MechanicalObject', name='learn_on_mo', tags="learnOn",
                                                    position=self.visible_surface_nodes, showObject=True,
                                                    showObjectScale=5, showColor=[1, 1, 1, 1])

        #  /root/mechanical/embedded_surface/rgbd_pcd
        rgbd_pcd = self.root.mechanical.embedded_surface.addChild('rgbd_pcd')
        self.rgbd_pcd_mo = rgbd_pcd.addObject('MechanicalObject', name='mo_rgbd_pcd',
                                              position=self.visible_surface_nodes, showObject=True,
                                              showObjectScale=5, showColor=[1, 0, 0, 1])  # translation=translation)
        rgbd_pcd.addObject('BarycentricMapping', input='@../../mo', output='@./')

        #  /root/mechanical/visual
        visual_node = self.root.mechanical.addChild('visual')
        self.visu = visual_node.addObject('OglModel', src='@../../surface_mesh', color='green', useVBO=True)
        visual_node.addObject('BarycentricMapping', input='@../mo', output='@./')

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
        self.regular_grid_rest_shape = from_sparse_to_regular_grid(self.nb_nodes_regular_grid, self.sparse_grid,
                                                                   self.behaviour_state)
        self.nb_nodes_sparse_grid = len(self.behaviour_state.rest_position.value)
        # Get the data sizes
        self.input_size = (self.nb_nodes_regular_grid, 1)
        self.output_size = (self.nb_nodes_regular_grid, 3)
        # Rendering
        self.initVisualizer()

    def initVisualizer(self):
        # Visualizer
        if self.visualizer is not None:
            self.visualizer.addMesh(positions=self.visu.position.value, cells=self.visu.triangles.value)

    def onAnimateBeginEvent(self, event):
        """
        Called within the Sofa pipeline at the beginning of the time step.
        :param event: Sofa Event
        :return: None
        """
        # Generate next forces
        p_force = self.config.p_force
        selected_centers = np.empty([0, 3])
        for j in range(p_force['nb_simultaneous_forces']):
            distance_check = True
            f = np.random.uniform(low=-1, high=1, size=(3,))
            f = (f / np.linalg.norm(f)) * p_force['amplitude_scale'] * np.random.random(1)

            # Pick up a random visible surface point and apply translation
            current_point = self.visible_surface_nodes[np.random.randint(len(self.visible_surface_nodes))]
            # current_point += np.array(self.config.p_liver['translation'])

            # Check if the current point is far enough from the already selected points
            for p in range(selected_centers.shape[0]):
                distance = np.linalg.norm(current_point - selected_centers[p])
                if distance < p_force['inter_distance_thresh']:
                    distance_check = False

            if distance_check:
                # Set the centers of the ROI sphere to current point
                selected_centers = np.concatenate((selected_centers, np.array([current_point])))
                self.spheres[j].centers.value = [current_point]

                # Build forces vector
                forces_vector = []
                for i in range(len(self.spheres[j].indices.array())):
                    forces_vector.append([f[0], f[1], f[2]])

                # Set forces and indices
                self.force_fields[j].indices.value = self.spheres[j].indices.array()
                self.force_fields[j].forces.value = forces_vector

            else:
                # If current point is too close to previously selected nodes, set forces to 0
                self.spheres[j].centers.value = [self.config.p_liver['fixed_point'].tolist()]
                self.force_fields[j].indices.value = [0]
                self.force_fields[j].forces.value = [[0.0, 0.0, 0.0]]

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

    def computeInput(self):
        """
        Compute the input to be given to the network. Automatically called by EnvironmentManager.
        :return: None
        """
        # Select random amount of visible surface
        indices = [random.randint(0, len(self.rgbd_pcd_mo.position.array()) - 1) for _ in
                   range(random.randint(50, 800))]
        indices = np.unique(np.array(indices))
        actual_positions_rgbd = self.rgbd_pcd_mo.position.array()[indices]
        # Initialize distance field
        DF_grid = np.zeros((self.nb_nodes_regular_grid, 1), dtype=np.double)
        # Get list of nodes of the cells containing a point from the RGBD point cloud and for each node,
        # compute distance to the considered point
        for p in actual_positions_rgbd:
            for node in self.grid.node_indices_of(self.grid.cell_index_containing(p)):
                if node < self.nb_nodes_regular_grid and DF_grid[node][0] == 0:
                    DF_grid[node] = np.linalg.norm(self.grid.node(node) - p)
        self.input = DF_grid

    def computeOutput(self):
        """
        Compute the output to be given to the network. Automatically called by EnvironmentManager.
        :return: None
        """
        actual_positions_on_regular_grid = np.zeros((self.nb_nodes_regular_grid, 3), dtype=np.double)
        actual_positions_on_regular_grid[self.idx_sparse_to_regular] = self.behaviour_state.position.array()
        self.output = copy.copy(np.subtract(actual_positions_on_regular_grid,
                                            self.regular_grid_rest_shape[None, :]))

    def checkSample(self, check_input=True, check_output=True):
        return self.converged

    def applyPrediction(self, prediction):
        """
        Apply the prediction of the network in the Sofa environment. Automatically called by EnvironmentManager.
        :return: None
        """
        # Needed for prediction only
        pass
