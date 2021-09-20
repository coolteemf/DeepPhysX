"""
Livers.py
Requirements : Sofa, Caribou
 - FEM simulated liver with random forces applied on the surface
 - NN simulated liver with predicted deformations given by UNet
"""

# Python imports
import os
import copy
import math
import numpy as np

# Sofa & Caribou imports
import Sofa.Simulation
import SofaRuntime
from Caribou.Topology import Grid3D

# DeepPhysX Sofa imports
from DeepPhysX_Sofa.Environment.SofaEnvironment import SofaEnvironment, BytesNumpyConverter, MeshVisualizer

# Working session imports
from Application.LiverRegistration.SofaScene.utils import from_sparse_to_regular_grid, extract_visible_nodes
from Application.LiverRegistration.SofaScene.parameters import p_liver, p_camera, p_grid, p_force


# Inherits from SofaEnvironment which allow to implement and create a Sofa scene in the DeepPhysX pipeline
class Livers(SofaEnvironment):

    def __init__(self, root_node, ip_address='localhost', port=10000, data_converter=BytesNumpyConverter,
                 instance_id=1, as_tcpip_client=True, visualizer_class=MeshVisualizer, environment_manager=None):
        SofaEnvironment.__init__(self, root_node=root_node, ip_address=ip_address, port=port,
                                 data_converter=data_converter, instance_id=instance_id,
                                 as_tcpip_client=as_tcpip_client, visualizer_class=visualizer_class,
                                 environment_manager=environment_manager)
        self.root = root_node
        # Keep a track of the actual step number and how many samples diverged during the animation
        self.nb_step = 0
        self.nb_converged = 0.
        self.converged = True
        # Flags set to True if each model is created
        self.is_created = {'fem': False, 'nn': False}
        self.F_obj = {}
        self.N_obj = {}

    def create(self):
        """
        Create the Sofa scene graph. Automatically called by SofaEnvironmentConfig.

        :return:
        """

        # Required plugins
        SofaRuntime.PluginRepository.addFirstPath(os.environ['CARIBOU_INSTALL'])
        required_plugins = ['SofaComponentAll', 'SofaLoader', 'SofaCaribou', 'SofaBaseTopology', 'SofaGeneralEngine',
                            'SofaEngine', 'SofaOpenglVisual', 'SofaBoundaryCondition']
        self.root.addObject('RequiredPlugin', pluginName=required_plugins)

        # Scene visual style
        self.root.addObject('VisualStyle', displayFlags="showVisualModels showBehaviorModels")

        # UNet regular grid
        self.regular_grid = Grid3D(anchor_position=p_grid['bbox_anchor'], n=p_grid['nb_cells'],
                                   size=p_grid['bbox_size'])
        print(f"3D grid: {self.regular_grid.number_of_nodes()} nodes, {self.regular_grid.number_of_cells()} cells.")

        # Liver model
        if '.obj' in p_liver['mesh_file']:
            self.surface_mesh = self.root.addObject('MeshObjLoader', name='surface_mesh', filename=p_liver['mesh_file'],
                                                    translation=p_liver['translation'])
        else:
            self.surface_mesh = self.root.addObject('MeshVTKLoader', name='surface_mesh', filename=p_liver['mesh_file'],
                                                    translation=p_liver[
                                                        'translation'])  # , scale3d=[10e-4,10e-4,10e-4])

        # Camera position
        self.root.addObject('MechanicalObject', name='camera', position=p_camera['camera_position'].tolist(),
                            showObject=True, showObjectScale=10)

        # Root children
        self.createModels()

    def createModels(self):
        """
        Small method to make inheritance easier.

        :return:
        """

        # Create the FEM simulated object
        self.createFEM()
        # Create the UNet simulated object
        # self.createNN()

    def createFEM(self):
        """
        FEM node of the liver scene

        :return:
        """

        # Create child node
        self.is_created['fem'] = True
        self.root.addChild('fem')

        # Surrounding box
        self.root.fem.addObject('BoxROI', box=p_grid['b_box'], drawBoxes=True, drawSize='1.0')

        # ODE solver + Static solver
        self.solver = self.root.fem.addObject('StaticODESolver', name='ODESolver', newton_iterations=10,
                                              printLog=False, correction_tolerance_threshold=1e-8,
                                              residual_tolerance_threshold=1e-8)
        self.root.fem.addObject('ConjugateGradientSolver', name='StaticSolver', preconditioning_method='Diagonal',
                                maximum_number_of_iterations=2000, residual_tolerance_threshold=1e-8, printLog=False)

        # Grid topology of the liver
        self.fem_sparse_grid_topo = self.root.fem.addObject('SparseGridTopology', name='sparse_grid_topo',
                                                            src='@../surface_mesh', n=p_grid['grid_resolution'])
        self.fem_sparse_grid_mo = self.root.fem.addObject('MechanicalObject', name='sparse_grid_mo',
                                                          src='@sparse_grid_topo', showObject=False)

        # Material
        self.root.fem.addObject('SaintVenantKirchhoffMaterial', name='stvk', young_modulus=5000, poisson_ratio=0.4)
        self.root.fem.addObject('HyperelasticForcefield', template="Hexahedron", material="@stvk",
                                topology='@sparse_grid_topo', printLog=True)

        # Fixed section of the liver
        self.root.fem.addObject('BoxROI', name='fixed_box', box=p_liver['fixed_box'], drawBoxes=True)
        self.root.fem.addObject('FixedConstraint', indices='@fixed_box.indices')

        # Surface child node
        self.root.fem.addChild('surface')
        self.fem_surface_topo = self.root.fem.surface.addObject('TriangleSetTopologyContainer', name='fem_surface_topo',
                                                                src='@../../surface_mesh')
        self.fem_surface_mo = self.root.fem.surface.addObject('MechanicalObject', name='fem_surface_mo',
                                                              src='@../../surface_mesh')
        self.root.fem.surface.addObject('BarycentricMapping', input='@../sparse_grid_mo', output='@./')

        # Forces
        self.sphere = []
        self.force_field = []
        for i in range(p_force['nb_simultaneous_forces']):
            self.sphere.append(self.root.fem.surface.addObject('SphereROI', name=f'sphere_{i}', radii=15,
                                                               drawSphere=True, drawSize=1,
                                                               centers=p_liver['fixed_point'].tolist()))
            self.force_field.append(self.root.fem.surface.addObject('ConstantForceField', name=f'cff_{i}', indices='0',
                                                                    forces=[0., 0., 0.], showArrowSize='0.2',
                                                                    showColor=[0, 0, 1, 1]))

        # Visible points
        cloud = self.root.fem.surface.addChild('points_cloud')
        self.points_cloud = cloud.addObject('MechanicalObject', name='visible_mo', position=[],
                                            showObject=True, showObjectScale=5, showColor=[1, 0, 0, 1])
        cloud.addObject('BarycentricMapping', input='@../../sparse_grid_mo', output='@./')

        # Visual
        self.root.fem.addChild('visual')
        self.fem_visu = self.root.fem.visual.addObject('OglModel', src='@../../surface_mesh', color='green')
        self.root.fem.visual.addObject('BarycentricMapping', input='@../sparse_grid_mo', output='@./')

    def createNN(self):
        """
        Neural Network node of the liver scene

        :return:
        """

        # Create child node
        self.is_created['nn'] = True
        self.root.addChild('nn')

        # Surrounding box
        self.root.nn.addObject('BoxROI', box=p_grid['b_box'], drawBoxes=True, drawSize='1.0')

        # Grid topology of the liver
        self.nn_sparse_grid_topo = self.root.nn.addObject('SparseGridTopology', name='sparse_grid_topo',
                                                          src='@../surface_mesh2', n=p_grid['grid_resolution'])
        self.nn_sparse_grid_mo = self.root.nn.addObject('MechanicalObject', name='sparse_grid_mo', showObject=False,
                                                        src='@sparse_grid_topo')

        # Fixed section of the liver
        self.root.nn.addObject('BoxROI', name='fixed_box', box=p_liver['fixed_box'], drawBoxes=True)

        # Surface child node
        self.root.nn.addChild('surface')
        self.nn_surface_mo = self.root.nn.surface.addObject('MechanicalObject', name='nn_surface_mo',
                                                            src='@../../surface_mesh2')
        self.nn_surface_topo = self.root.nn.surface.addObject('TriangleSetTopologyContainer', name='triangle_topo',
                                                              src='@../../surface_mesh2')
        self.root.nn.surface.addObject('BarycentricMapping', input='@../sparse_grid_mo', output='@./')

        # Forces
        if not self.is_created['fem']:
            self.sphere = self.root.nn.surface.addObject('SphereROI', name='sphere', radii=0.015, drawSphere=True,
                                                         centers=p_liver['fixed_point'].tolist(), drawSize=1)
            self.force_field = self.root.nn.surface.addObject('ConstantForceField', name='cff', indices='0',
                                                              forces=[0., 0., 0.], showArrowSize='0.2',
                                                              showColor=[1, 0, 1, 1])

        # Visible points
        if not self.is_created['fem']:
            cloud = self.root.fem.fem_surface.addChild('points_cloud')
            self.points_cloud = cloud.addObject('MechanicalObject', name='visible_mo', position=self.visible_nodes,
                                                showObject=True, showObjectScale=5, showColor=[1, 0, 0, 1])
            cloud.addObject('BarycentricMapping', input='@../../sparse_grid_mo', output='@./')

        # Visual
        self.root.nn.addChild('visual')
        self.nn_visu = self.root.nn.visual.addObject('OglModel', src='@../../surface_mesh2', color='orange')
        self.root.nn.visual.addObject('BarycentricMapping', input='@../sparse_grid_mo', output='@./')

    def onSimulationInitDoneEvent(self, event):
        """
        Called within the Sofa pipeline at the end of the scene graph initialisation.

        :param event: Sofa Event
        :return: None
        """

        # Correspondences between sparse grid and regular grid
        sparse_grid_mo = self.fem_sparse_grid_mo if self.is_created['fem'] else self.nn_sparse_grid_mo
        sparse_grid_topo = self.fem_sparse_grid_topo if self.is_created['fem'] else self.nn_sparse_grid_topo
        grid_shape = p_grid['grid_resolution']
        self.nb_nodes_regular_grid = grid_shape[0] * grid_shape[1] * grid_shape[2]
        self.nb_nodes_sparse_grid = len(sparse_grid_mo.rest_position.value)
        sparse_to_regular_stuff = from_sparse_to_regular_grid(self.nb_nodes_regular_grid, sparse_grid_topo,
                                                              sparse_grid_mo)
        self.idx_sparse_to_regular, self.idx_regular_to_sparse = sparse_to_regular_stuff[0], sparse_to_regular_stuff[1]
        self.regular_grid_rest_shape = sparse_to_regular_stuff[2]

        # Save the edges between nodes to compute occlusions later
        self.neighboors = [[] for _ in range(self.fem_surface_topo.nbPoints.value)]
        for e in self.fem_surface_topo.edges.value:
            self.neighboors[e[0]].append(e[1])
            self.neighboors[e[1]].append(e[0])

        idx_visible_nodes, _ = extract_visible_nodes(camera_position=p_camera['camera_position'],
                                                     object_position=p_liver['fixed_point'],
                                                     normals=self.fem_visu.normal.value,
                                                     positions=self.fem_surface_mo.position.value,
                                                     dot_thresh=p_camera['camera_thresholds'][0],
                                                     distance_from_camera_thresh=p_camera['camera_thresholds'][1],
                                                     inversed_normals=True)
        position = self.fem_surface_mo.position.value[np.array(idx_visible_nodes)].tolist()
        self.root.fem.surface.points_cloud.removeObject(self.points_cloud)
        self.points_cloud = self.root.fem.surface.points_cloud.addObject('MechanicalObject', name='visible_mo',
                                                                         position=position, showObject=True,
                                                                         showObjectScale=5, showColor=[1, 0, 0, 1])
        self.points_cloud.init()

        # Get the data sizes
        self.input_size = (self.nb_nodes_regular_grid, 1)
        self.output_size = (self.nb_nodes_regular_grid, 3)

    def send_visualization(self):
        """
        Call by TcpIpClient when init is done to send data to TciIpServer
        :return:
        """
        visu_dict = {}
        visu_dict = self.visualizer.createObjectData(data_dict=visu_dict, positions=self.fem_visu.position.value,
                                                     cells=self.fem_visu.triangles.value, at=0)
        translation = np.array(p_liver['nn_translation'] * self.nn_visu.position.shape[0]).reshape(
            self.nn_visu.position.shape)
        visu_dict = self.visualizer.createObjectData(data_dict=visu_dict,
                                                     positions=self.nn_visu.position.value + translation,
                                                     cells=self.nn_visu.triangles.value, at=0)
        return visu_dict

    def onAnimateBeginEvent(self, event):
        """
        Called within the Sofa pipeline at the beginning of the time step.

        :param event: Sofa Event
        :return: None
        """
        # Reset positions
        if self.is_created['fem']:
            self.fem_sparse_grid_mo.position.value = self.fem_sparse_grid_mo.rest_position.value

        # Build and set forces vectors
        selected_centers = np.empty([0, 3])
        for i in range(p_force['nb_simultaneous_forces']):
            # Pick up a random visible surface point, select the points in a centered sphere
            current_point = self.fem_surface_mo.position.value[np.random.randint(len(self.fem_surface_mo.position.value))]
            # Check distance to other points
            distance_check = True
            for p in selected_centers:
                distance = np.linalg.norm(current_point - p)
                if distance < p_force['inter_distance_thresh']:
                    distance_check = False
                    break
            empty_indices = False
            if distance_check:
                # Add center to the selection
                selected_centers = np.concatenate((selected_centers, np.array([current_point])))
                # Set sphere center
                self.sphere[i].centers.value = [current_point]
                # Build force vector
                if len(self.sphere[i].indices.value) > 0:
                    f = np.random.uniform(low=-1, high=1, size=(3,))
                    f = (f / np.linalg.norm(f)) * p_force['amplitude_scale'] * np.random.random(1)
                    forces_vector = [f for _ in range(len(self.sphere[i].indices.value))]
                    self.force_field[i].indices.value = self.sphere[i].indices.array()
                    self.force_field[i].forces.value = forces_vector
                    self.force_field[i].showArrowSize.value = 10 / np.linalg.norm(f)
                else:
                    empty_indices = True
            if not distance_check or empty_indices:
                # Reset sphere position
                self.sphere[i].centers.value = [p_liver['fixed_point'].tolist()]
                # Reset force field
                self.force_field[i].indices.value = [0]
                self.force_field[i].forces.value = [[0.0, 0.0, 0.0]]

    def onAnimateEndEvent(self, event):
        """
        Called within the Sofa pipeline at the end of the time step.

        :param event: Sofa Event
        :return: None
        """
        # Check whether if the solver diverged or not
        self.checkSample()

        # Extract visible nodes
        idx_visible_nodes, _ = extract_visible_nodes(camera_position=p_camera['camera_position'],
                                                     object_position=p_liver['fixed_point'],
                                                     normals=self.fem_visu.normal.value,
                                                     positions=self.fem_surface_mo.position.value,
                                                     dot_thresh=p_camera['camera_thresholds'][0],
                                                     distance_from_camera_thresh=p_camera['camera_thresholds'][1],
                                                     inversed_normals=True)

        # Compute occlusions and noise
        idx_visible_nodes = self.computeOcclusions(idx_visible_nodes)
        visible_nodes = self.fem_surface_mo.position.value[np.array(idx_visible_nodes)]
        visible_nodes = self.computeNoise(visible_nodes)

        # Update visible points MO
        position = self.fem_surface_mo.position.value[np.array(idx_visible_nodes)].tolist()
        self.root.fem.surface.points_cloud.removeObject(self.points_cloud)
        self.points_cloud = self.root.fem.surface.points_cloud.addObject('MechanicalObject', name='visible_mo',
                                                                         position=position, showObject=True,
                                                                         showObjectScale=5, showColor=[1, 0, 0, 1])
        self.points_cloud.init()

        # Send training data
        # if self.as_tcpip_client:
        #     self.sync_send_training_data(network_input=self.computeInput(),
        #                                  network_output=self.computeOutput())
        #     # self.update_visualization()
        #     self.sync_send_command_done()
        # self.setTrainingData(input_array=self.computeInput(), output_array=self.computeOutput())

        self.nb_step += 1

    def computeOcclusions(self, idx_visible_nodes):
        """

        :param idx_visible_nodes:
        :return:
        """
        # Number of occlusion randomly chosen with exponential distribution
        d = np.random.exponential()
        nb_occlusion = 0
        for i in range(p_camera['max_occlusions'], 0, -1):
            if d <= math.exp(-i):
                nb_occlusion = i
                break

        # Randomly process occlusions neighbor by neighbor
        for _ in range(nb_occlusion):
            seed = idx_visible_nodes[np.random.randint(len(idx_visible_nodes))]
            [min_prop, max_prop] = [*p_camera['min_max_occlusion_proportions']]
            size = np.random.randint(min_prop * len(idx_visible_nodes), max_prop * len(idx_visible_nodes))
            nodes_to_remove = [seed]
            # Remove nodes until reaching the defined size or until their is no more neighbors to remove
            for _ in range(size):
                if len(nodes_to_remove) == 0:
                    break
                node = nodes_to_remove.pop(0)
                idx_visible_nodes.remove(node)
                for n in self.neighboors[node]:
                    if n not in nodes_to_remove and n in idx_visible_nodes:
                        nodes_to_remove.append(n)

        return idx_visible_nodes

    def computeNoise(self, visible_nodes):
        """

        :param visible_nodes:
        :return:
        """
        noise = np.random.normal(0, p_camera['noise_level'], visible_nodes.shape)
        return visible_nodes + noise

    def computeInput(self):
        """
        Compute the input to be given to the network.

        :return: None
        """

        # Select a random amount of points on the visible point cloud
        # indices = [np.random.randint(0, len(self.points_cloud.position.array()) - 1)
        #            for _ in range(np.random.randint(50, 800))]
        # indices = np.unique(np.array(indices))
        # actual_positions_of_point_cloud = self.points_cloud.position.array()[indices]
        actual_positions_of_point_cloud = self.points_cloud.position.array()

        # Encoding displacement field
        # Init distance field to zero
        DF_grid = np.zeros((self.nb_nodes_regular_grid, 1))
        # Get the list of nodes composing a cell containing a point from the visible point cloud
        # For each node of the cell, compute the distance to the considered point
        for p in actual_positions_of_point_cloud:
            cell = self.regular_grid.cell_index_containing(p)
            for node in self.regular_grid.node_indices_of(cell):
                if node < self.nb_nodes_regular_grid:
                    dist = np.linalg.norm(self.regular_grid.node(node) - p)
                    if DF_grid[node] == 0 or dist < DF_grid[node]:
                        DF_grid[node] = dist

        return DF_grid

    def computeOutput(self):
        """
        Compute the output to be given to the network.

        :return: None
        """

        # Write the position of each point from the sparse grid to the regular grid
        actual_positions_on_regular_grid = np.zeros((self.nb_nodes_regular_grid, 3), dtype=np.double)
        actual_positions_on_regular_grid[self.idx_sparse_to_regular] = self.fem_sparse_grid_mo.position.array()
        return copy.copy(np.subtract(actual_positions_on_regular_grid, self.regular_grid_rest_shape))

    def checkSample(self, check_input=True, check_output=True):
        """
        Check if the produced sample is correct. Automatically called by EnvironmentManager.

        :param check_input:
        :param check_output:
        :return:
        """
        if self.is_created['fem']:
            self.converged = self.solver.converged.value
            if not self.converged:
                Sofa.Simulation.reset(self.root)
            self.nb_converged += int(self.converged)
        return self.converged

    def applyPrediction(self, prediction):
        """
        Apply the prediction of the network in the Sofa environment. Automatically called by EnvironmentManager.

        :return: None
        """
        # Reshape to correspond regular grid, transform to sparse grid
        U = np.reshape(prediction, (self.nb_nodes_regular_grid, 3))
        U_sparse = U[self.idx_sparse_to_regular]

        # Apply displacement to the sparse grid, mapping will impose surface displacement
        self.nn_sparse_grid_mo.position.value = self.nn_sparse_grid_mo.rest_position.array() + U_sparse
        # self.compute_error()

        visu_dict = {}
        visu_dict = self.visualizer.updateObjectData(data_dict=visu_dict, positions=self.fem_visu.position.value)
        translation = np.array(p_liver['nn_translation'] * self.nn_visu.position.shape[0]).reshape(
            self.nn_visu.position.shape)
        visu_dict = self.visualizer.updateObjectData(data_dict=visu_dict,
                                                     positions=self.nn_visu.position.value + translation)
        self.setVisualizationData(visu_dict)

    def compute_error(self):
        """

        :return:
        """
        pred = self.nn_sparse_grid_mo.position.array() - self.nn_sparse_grid_mo.rest_position.array()
        gt = self.fem_sparse_grid_mo.position.array() - self.fem_sparse_grid_mo.rest_position.array()
        print(1000 * np.max(pred - gt), 1000 * np.linalg.norm(pred - gt))

    def close(self):
        """
        Ending method of Environment. Automatically called by EnvironmentManager.

        :return:
        """
        pass
