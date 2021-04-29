import random
import numpy as np
from Caribou.Topology import Grid3D

from DeepPhysX_Sofa.Environment.SofaBaseEnvironment import SofaBaseEnvironment
from utils import extract_visible_nodes, from_sparse_to_regular_grid


class LiverEnvironment(SofaBaseEnvironment):

    def __init__(self, root_node, config, idx_instance=1):
        super(LiverEnvironment, self).__init__(root_node, config, idx_instance)
        self.config = config

    def create(self, config):

        # Get parameters
        p_liver = config.p_liver
        p_grid = config.p_grid
        p_forces = config.p_forces

        # /root
        surface_mesh_loader = self.rootNode.addObject('MeshObjLoader', name='surface_mesh',
                                                      filename=p_liver['mesh_file'], translation=p_liver['translation'])
        self.visible_surface_nodes = extract_visible_nodes(camera_position=p_liver['camera_position'],
                                                           normals=surface_mesh_loader.normals.value,
                                                           positions=surface_mesh_loader.position.value,
                                                           dot_thresh=0.0, rand_thresh=0.0,
                                                           distance_from_camera_thresh=1e6)
        self.grid = Grid3D(anchor_position=p_grid['min_bbox'], n=p_grid['nb_cells'], size=p_grid['bbox_size'])
        print("3D grid: {} nodes, {} cells.".format(self.grid.number_of_nodes(), self.grid.number_of_cells()))

        # /root/mechanical_node
        mechanical_node = self.rootNode.addChild('mechanical_node')
        self.solver = mechanical_node.addObject('LegacyStaticODESolver', newton_iterations=10,
                                                correction_tolerance_threshold=1e-6,
                                                residual_tolerance_threshold=1e-6,
                                                printLog=False)
        mechanical_node.addObject('ConjugateGradientSolver', name='CGSolverCaribou', preconditioning_method='Diagonal',
                                  maximum_number_of_iterations=2000, residual_tolerance_threshold=1e-9,
                                  printLog=False)
        self.sparseGrid = mechanical_node.addObject('SparseGridTopology', name='sparse_grid', src='@../surface_mesh',
                                                    n=p_grid['grid_resolution'])
        mechanical_node.addObject('BoxROI', name='bbox', box=[p_grid['min_bbox'], p_grid['max_bbox']],
                                  drawBoxes=True, drawSize='1.0')
        self.behaviour_state = mechanical_node.addObject('MechanicalObject', name='mo', src='@sparse_grid',
                                                         showObject=False)
        mechanical_node.addObject('SaintVenantKirchhoffMaterial', name="stvk", young_modulus=5000, poisson_ratio=0.4)
        mechanical_node.addObject('HyperelasticForcefield', template="Hexahedron", material="@stvk", printLog=True)
        mechanical_node.addObject('BoxROI', name='fixed_box_roi', box=p_liver['fixed_box'], drawBoxes=True)
        mechanical_node.addObject('FixedConstraint', indices='@fixed_box_roi.indices')

        # /root/mechanical_node/embedded_surface
        embedded_surface = mechanical_node.addChild('embedded_surface')
        embedded_surface.addObject('MechanicalObject', name='mo_embedded_surface', src='@../../surface_mesh')
        embedded_surface.addObject('TriangleSetTopologyContainer', name='triangleTopo', src='@../../surface_mesh')
        nb_forces = p_forces['nb_simultaneous_forces']
        self.spheres = [embedded_surface.addObject('SphereROI', name='sphere' + str(i), tags='ROI_SPHERE' + str(i),
                                                       centers=p_liver['fixed_point'].tolist(), radii=0.015,
                                                       drawSphere=True, drawSize=1) for i in range(nb_forces)]
        self.force_field = [embedded_surface.addObject('ConstantForceField', name='cff' + str(i), tags='CFF' + str(i),
                                                   indices='0', forces=[0., 0., 0.], showArrowSize='0.2',
                                                   showColor=[1, 0, 1, 1]) for i in range(nb_forces)]
        embedded_surface.addObject('BarycentricMapping', input='@../mo', output='@./')

        # /root/mechanical_node/embedded_surface/visible_points
        visible_points = embedded_surface.addChild('visible_points')
        self.learn_on_mo = visible_points.addObject('MechanicalObject', name='learn_on_mo', tags="learnOn",
                                                    position=self.visible_surface_nodes, showObject=True,
                                                    showObjectScale=5, showColor=[0, 1, 1, 1])

        #  /root/mechanical_node/embedded_surface/rgbd_pcd
        rgbd_pcd = embedded_surface.addChild('rgbd_pcd')
        self.rgbd_pcd_mo = rgbd_pcd.addObject('MechanicalObject', name='mo_rgbd_pcd',
                                              position=self.visible_surface_nodes, showObject=True,
                                              showObjectScale=5, showColor=[1, 0, 0, 1])  # translation=translation)
        rgbd_pcd.addObject('BarycentricMapping', input='@../../mo', output='@./')

        #  /root/mechanical_node/visual
        visual_node = mechanical_node.addChild('visual')
        visual_node.addObject('OglModel', src='@../../surface_mesh', color='green')
        visual_node.addObject('BarycentricMapping', input='@../mo', output='@./')
        print("done")

        return self.rootNode

    def onSimulationInitDoneEvent(self, event):
        grid_shape = self.config.p_grid['grid_resolution']
        self.nb_nodes_regular_grid = grid_shape[0] * grid_shape[1] * grid_shape[2]
        # Initialize inputs and outputs
        self.indices_sparse_to_regular, \
        self.indices_regular_to_sparse, \
        self.regular_grid_rest_shape = from_sparse_to_regular_grid(self.nb_nodes_regular_grid, self.sparseGrid,
                                                                   self.behaviour_state)
        self.nb_nodes_sparse_grid = len(self.behaviour_state.rest_position.value)
        print("Number of nodes in sparse grid is {}".format(self.nb_nodes_sparse_grid))
        print("Number of nodes in regular grid is {}".format(self.nb_nodes_regular_grid))
        self.inputs = np.empty((0, *(self.nb_nodes_regular_grid, 1)))
        self.outputs = np.empty((0, *(self.nb_nodes_regular_grid, 3)))

    def onStep(self):
        self.computeInput()
        self.computeOutput()

    def checkSample(self):
        pass

    def computeInput(self):
        # Select random amount of visible surface
        indices = [random.randint(0, len(self.rgbd_pcd_mo.position.array()) - 1) for _ in range(random.randint(50, 800))]
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
        self.inputs = np.concatenate((self.inputs, DF_grid[None, :]))

    def computeOutput(self):
        actual_positions_regular_grid = np.zeros((self.nb_nodes_regular_grid, 3), dtype=np.double)
        actual_positions_regular_grid[self.indices_sparse_to_regular] = self.behaviour_state.position.array()
        self.outputs = np.concatenate((self.outputs, np.copy(np.subtract(actual_positions_regular_grid,
                                                                         self.regular_grid_rest_shape))[None, :]))

    def onAnimateEndEvent(self, event):
        # Generate next forces
        selected_centers = np.empty([0, 3])
        for i in range(self.config.p_forces['nb_simultaneous_forces']):
            inter_distance_check = True
            F = np.random.uniform(low=-1, high=1, size=3)
            F = (F / np.linalg.norm(F)) * self.config.p_forces['amplitude_scale'] * np.random.random(1)
            # Pick a random visible surface point and apply translation
            current_point = self.visible_surface_nodes[np.random.randint(len(self.visible_surface_nodes))]
            # Check if the current point is far enough from the already selected points
            for p in range(selected_centers.shape[0]):
                distance = np.linalg.norm(current_point - selected_centers[p])
                inter_distance_check = distance > self.config.p_forces['inter_distance_thresh']
                if not inter_distance_check:
                    break
            if inter_distance_check:
                # Set the center of the ROI spheres
                selected_centers = np.concatenate((selected_centers, np.array([current_point])))
                self.spheres[i].centers.value = [current_point]
                # Build force vector
                force_vector = []
                for j in range(len(self.spheres[i].indices.array())):
                    force_vector.append(F.tolist())
                # Set forces and indices
                self.force_field[i].indices.value = self.spheres[i].indices.array()
                self.force_field[i].forces.value = force_vector
            else:
                # If current point is too close set force to 0
                self.spheres[i].centers.value = [self.config.p_liver['fixed_point'].tolist()]
                self.force_field[i].indices.value = [0]
                self.force_field[i].forces.value = [[0., 0., 0.]]
