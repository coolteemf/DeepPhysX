# Python imports
import os
import numpy as np
import copy
import random
import time
from time import clock_gettime_ns as timer
from torch.optim import Adam
from torch.nn.functional import mse_loss

# Sofa imports
import Sofa.Gui
import SofaRuntime
import Sofa.Simulation
from Caribou.Topology import Grid3D
from Resources import sparseGridUtils as sgu
from Resources.Trainer import Unet_Training
import NeuralNetwork.unet.network as unet

meshfile = "/home/robin/dev/kromagnon-master/src/Data/mesh_kromagnon/liver/patient3/patient3_parenchyme_aligned_and_trimmed_simplified.obj"
network_name = "patient3_test"
path_to_save = "../../Data/" + network_name

# Transformation to center the liver
translation = [-0.0134716, 0.021525, -0.427]

# Fixed constraints for centered liver
fixed_point = np.array([-0.00338, -0.0256, 0.52]) + np.array(translation)
fixed_width = np.array([0.07, 0.05, 0.04])
fixed_box = (fixed_point - fixed_width / 2.).tolist() + (fixed_point + fixed_width / 2.).tolist()

# # # Compute grid resolution for desired cell size and bbox size centered at (0, 0, 0)
# grid_size = 20
# bbox_size = 0.3
# minbb = [- bbox_size * 0.5, - bbox_size * 0.5, - bbox_size * 0.5]
# maxbb = [bbox_size * 0.5, bbox_size * 0.5, bbox_size * 0.5]
# grid_resolution = [grid_size, grid_size, grid_size]
# print("Grid resolution is {}".format(grid_resolution))

cell_size = 0.07
margins = np.array([0.02, 0.02, 0.02])
minbb = np.array([-0.130815, -0.107192, 0.00732511]) - margins
maxbb = np.array([0.0544588, 0.0967464, 0.15144]) + margins
grid_resolution = sgu.compute_grid_resolution(maxbb, minbb, cell_size)
print("Grid resolution is {}".format(grid_resolution))
bbox_size = maxbb - minbb
minbb = minbb.tolist()
maxbb = maxbb.tolist()

nb_cells_x = grid_resolution[0] - 1
nb_cells_y = grid_resolution[1] - 1
nb_cells_z = grid_resolution[2] - 1


class Liver(Unet_Training.Unet_Training):
    def __init__(self, node, Loop, *args, **kwargs):
        Unet_Training.Unet_Training.__init__(self, node, Loop,
                                             scenarioName=network_name,
                                             networkName=network_name,
                                             # dataset="../../Training/Data/input_DF_test_only_cvg_samples/dataset/",
                                             partition_size=0.2,
                                             saveTestData=False, nbTestsToSave=1,
                                             shuffle=True,
                                             dataScale=1000.0,
                                             gridShape=(grid_resolution[2], grid_resolution[1], grid_resolution[0], 1),
                                             loss_threshold=1000.0,
                                             keep_losses=False,
                                             multiple_partitions=False,
                                             *args, **kwargs)
        self.forcefield = None
        self.solver = None
        self.sparsegrid = None
        self.behavior_state = None
        self.indices_of_sparsegrid_in_regulargrid = None
        self.indices_of_regulargrid_in_sparsegrid = None
        self.nb_nodes_sparse_grid = None
        self.nb_nodes_regular_grid = None
        self.regular_grid_rest_shape_position = None
        self.visible_nodes_in_sparse_grid = None
        self.visible_nodes_in_regular_grid = None
        self.visible_nodes_regular_grid_rest_shape_position = None
        self.embedded_surface_mesh = None
        self.spheres = None
        self.force_fields = None
        self.learnOn_mo = None
        self.rgbd_pcd_mo = None

        self.nb_nodes_regular_grid = self.gridShape[2] * self.gridShape[1] * self.gridShape[0]
        self.nb_simultaneous_forces = nbSimultaneousForces
        # self.manager.datasetManager.input_shape = [grid_resolution[2], grid_resolution[1], grid_resolution[0], 1]
        # self.manager.datasetManager.output_shape = [grid_resolution[2], grid_resolution[1], grid_resolution[0], 3]

        config = unet.UNetConfig(
            steps=3,
            first_layer_channels=128,
            num_classes=3,
            num_input_channels=self.gridShape[3],
            ndims=3,
            border_mode='same',
            two_sublayers=True
        )
        self.manager.networkManager.setNetwork(config,
                                               network=unet.UNet,
                                               optimizer=Adam,
                                               lr=1e-4,
                                               loss=mse_loss)
        self.createGraph(node)

    def getInput(self, increment):
        increment.value += 1
        # start = timer(time.CLOCK_REALTIME)
        # Select random amount of visible surface
        indices = [random.randint(0, len(self.rgbd_pcd_mo.position.array()) - 1) for i in
                   range(random.randint(50, 800))]
        indices = np.unique(np.array(indices))
        actual_positions_of_rgbd_pcd = self.rgbd_pcd_mo.position.array()[indices]

        # Initialize distance field to 0
        DF_grid = np.zeros((self.nb_nodes_regular_grid, 1), dtype=np.double)

        # Get list of nodes of the cells containing a point from the RGBD point cloud and
        # for each node, compute the distance to the considered point
        for p in actual_positions_of_rgbd_pcd:
            for node in self.grid_df.node_indices_of(self.grid_df.cell_index_containing(p)):
                if node < self.nb_nodes_regular_grid and DF_grid[node][0] == 0:
                    DF_grid[node] = np.linalg.norm(self.grid_df.node(node) - p)

        # end = timer(time.CLOCK_REALTIME)
        # print("Voxelization using Caribou grids took {} milliseconds.".format((end-start)*1e-6))
        self.inputs = np.concatenate((self.inputs, DF_grid[None, :]))

    def getOutput(self, increment):
        actual_positions_on_regular_grid = np.zeros((self.nb_nodes_regular_grid, 3), dtype=np.double)
        actual_positions_on_regular_grid[
            self.indices_of_sparsegrid_in_regulargrid] = self.behavior_state.position.array()
        self.outputs = np.concatenate(
            (self.outputs, copy.copy(np.subtract(actual_positions_on_regular_grid, self.regular_grid_rest_shape_position[None, :]))))
        self.rootNode.Loop.currentBatchSize = len(self.outputs)

    def createGraph(self, rootNode):
        surface_mesh_loader = rootNode.addObject(
            'MeshObjLoader',
            name='surface_mesh',
            filename=meshfile,
            translation=translation
        )
        self.visible_surface_nodes_position = sgu.extract_visible_mo(
            camera_position=np.array([-0.177458, 0.232606, 0.780813]),
            normals=surface_mesh_loader.normals.value,
            positions=surface_mesh_loader.position.value,
            dot_threshold=0.0, rand_threshold=0.0, distance_from_camera_threshold=1e6)

        # Grid3D from caribou in order to encode the RGB-D point cloud as a distance field
        self.grid_df = Grid3D(anchor_position=minbb, n=[nb_cells_x, nb_cells_y, nb_cells_z], size=bbox_size)
        print("Number of nodes in grid:", self.grid_df.number_of_nodes())
        print("Number of cells in grid:", self.grid_df.number_of_cells())

        #  /root/mechanical_node/
        mechanical_node = rootNode.addChild('mechanical')
        self.solver = mechanical_node.addObject(
            'StaticODESolver',
            newton_iterations=10,
            correction_tolerance_threshold=1e-6,
            residual_tolerance_threshold=1e-6,
            shoud_diverge_when_residual_is_growing=False,
            printLog=False
        )
        mechanical_node.addObject(
            'ConjugateGradientSolver',
            name='CGSolver',
            preconditioning_method='Diagonal',
            maximum_number_of_iterations=2000,
            residual_tolerance_threshold=1.e-9,
            printLog=False
        )
        # Sparse Grid for FEM computations
        self.sparsegrid = mechanical_node.addObject(
            'SparseGridTopology',
            name='sparsegrid',
            min=minbb,
            max=maxbb,
            n=[self.gridShape[2], self.gridShape[1], self.gridShape[0]],
            src='@../surface_mesh'
        )
        # Bbox for visual checking
        mechanical_node.addObject('BoxROI', box=[minbb, maxbb], name='bbox', drawBoxes=True)

        # Mechanical object
        self.behavior_state = mechanical_node.addObject(
            'MechanicalObject',
            src='@sparsegrid',
            name='mo',
            showObject=False
        )
        # Saint Venant Kirchoff material and Force Field
        mechanical_node.addObject(
            'SaintVenantKirchhoffMaterial',
            young_modulus=5000,
            poisson_ratio=0.4,
            name="stvk"
        )
        mechanical_node.addObject(
            'HyperelasticForcefield',
            material="@stvk",
            template="Hexahedron",
            printLog=True
        )
        # Fixed boundary conditions
        mechanical_node.addObject('BoxROI', box=fixed_box, name='fixed_box_roi', drawBoxes=True)
        mechanical_node.addObject('FixedConstraint', indices='@fixed_box_roi.indices')

        #  /root/mechanical_node/embedded_surface
        embedded_surface = mechanical_node.addChild('embedded_surface')
        embedded_surface.addObject('MechanicalObject', name='mo_embedded_surface', src='@../../surface_mesh')
        embedded_surface.addObject('TriangleSetTopologyContainer', name="triangleTopo", src="@../../surface_mesh")

        self.force_fields = []
        self.spheres = []
        for i in range(self.nb_simultaneous_forces):
            self.spheres.append(
                embedded_surface.addObject('SphereROI', tags='ROI_sphere' + str(i), name='sphere' + str(i),
                                           centers=fixed_point.tolist(), radii=0.015, drawSphere=True))
            self.force_fields.append(
                embedded_surface.addObject('ConstantForceField', tags='CFF' + str(i), name='cff' + str(i), indices='0',
                                           forces=[0.0, 0.0, 0.0], showArrowSize='0.002'))
        embedded_surface.addObject('BarycentricMapping', input='@../mo', output='@./')

        #  /root/mechanical_node/embedded_surface/visible_points
        visible_points = embedded_surface.addChild('visible_points')
        self.learnOn_mo = visible_points.addObject('MechanicalObject', tags="learnOn", name='mo_learnOn',
                                                   position=self.visible_surface_nodes_position,
                                                   showObject=True,
                                                   showObjectScale=5,
                                                   showColor=[0, 1, 1, 1])  # translation=translation)

        #  /root/mechanical_node/embedded_surface/rgbd_pcd
        rgbd_pcd = embedded_surface.addChild('rgbd_pcd')
        self.rgbd_pcd_mo = rgbd_pcd.addObject('MechanicalObject', name='mo_rgbd_pcd',
                                              position=self.visible_surface_nodes_position,
                                              showObject=True, showObjectScale=5, showColor=[1, 0, 0, 1])  # ,
        # translation=translation)
        rgbd_pcd.addObject('BarycentricMapping', input='@../../mo', output='@./')

        #  /root/mechanical_node/visual
        visual_node = mechanical_node.addChild('visual')
        visual_node.addObject('OglModel', src='@../../surface_mesh', color='green')
        visual_node.addObject('BarycentricMapping', input='@../mo', output='@./')

    def onSimulationInitDoneEvent(self, value):
        print("onSimulationInitDoneEvent")
        # Initialize mapping between sparse grid and regular grid
        self.indices_of_sparsegrid_in_regulargrid, \
        self.indices_of_regulargrid_in_sparsegrid, \
        self.regular_grid_rest_shape_position = sgu.from_sparse_to_regular_grid(
            self.nb_nodes_regular_grid,
            self.sparsegrid,
            self.behavior_state)
        self.nb_nodes_sparse_grid = len(self.behavior_state.rest_position.value)
        print("Number of nodes in sparse grid is {}".format(self.nb_nodes_sparse_grid))
        print("Number of nodes in regular grid is {}".format(self.nb_nodes_regular_grid))
        self.inputs = np.empty((0, *(self.nb_nodes_regular_grid, 1)))
        self.outputs = np.empty((0, *(self.nb_nodes_regular_grid, 3)))

    def onAnimateEndEvent(self, value):
        Unet_Training.Unet_Training.onAnimateEndEvent(self, value)

        # Generate next forces
        if self.manager.datasetManager.mode == "online_training":
            selected_centers = np.empty([0, 3])
            for j in range(nbSimultaneousForces):
                distance_check = True
                f = np.random.uniform(low=-1, high=1, size=(3,))
                f = (f / np.linalg.norm(f)) * amplitudeScale * np.random.random(1)

                # Pick up a random visible surface point and apply translation
                current_point = self.visible_surface_nodes_position[
                    np.random.randint(len(self.visible_surface_nodes_position))]
                # current_point += np.array(translation)

                # Check if the current point is far enough from the already selected points
                for p in range(selected_centers.shape[0]):
                    distance = np.linalg.norm(current_point - selected_centers[p])
                    if distance < thresholdDistance:
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
                    self.spheres[j].centers.value = [fixed_point.tolist()]
                    self.force_fields[j].indices.value = [0]
                    self.force_fields[j].forces.value = [[0.0, 0.0, 0.0]]


# Forces related variables
amplitudeSample = 10
amplitudeScale = 0.05
sphereSample = 2000
sphereSampleStep = 1
nbSimultaneousForces = 20
nbSamples = 2000
thresholdDistance = 0.06
ROISphereCenter = str(fixed_point)[1:-1]
# Training related variables
batch_size = 5
nb_epoch = 100
batch_per_epoch = 30


def createScene(rootNode):
    Loop = rootNode.addObject('TrainingAnimationLoop', name="Loop", load_dataset=False, always_apply_physics=False,
                              batch_size=batch_size, batch_per_epoch=batch_per_epoch, total_epochs=nb_epoch,
                              test_every=-1, simulation_per_step=1)
    rootNode.addObject('AttachBodyButtonSetting', name='mouse', stiffness='1.0')
    rootNode.addObject('DefaultPipeline', depth='6', verbose='0', draw='0', name='DefaultCollisionPipeline')
    rootNode.addObject('BruteForceDetection', name='Detection')
    rootNode.addObject('DiscreteIntersection', name='Intersection')
    rootNode.addObject('ViewerSetting', name='viewer')
    rootNode.addObject('BackgroundSetting', name='ViewerBGcolor', color='0.1 0.1 0.1')
    rootNode.addObject('VisualStyle', displayFlags='showVisualModels showBehaviorModels showForceFields showWireframe')
    rootNode.addObject(Liver(rootNode, Loop))
    return rootNode


if __name__ == "__main__":
    # Add includes
    SofaRuntime.PluginRepository.addFirstPath('/home/robin/dev/SofaPython3/build/lib')
    SofaRuntime.PluginRepository.addFirstPath('/home/robin/dev/sofa/build/lib')
    SofaRuntime.PluginRepository.addFirstPath('/home/robin/dev/caribou/build/lib')
    SofaRuntime.PluginRepository.addFirstPath('/home/robin/dev/kromagnon-master/build/lib')

    # # Register all the common component in the factory.
    SofaRuntime.importPlugin("SofaComponentAll")
    SofaRuntime.importPlugin("SofaOpenglVisual")
    SofaRuntime.importPlugin("SofaPython3")
    SofaRuntime.importPlugin("SofaCaribou")
    SofaRuntime.importPlugin("SofaKromagnon")
    SofaRuntime.importPlugin("SofaLoader")
    SofaRuntime.importPlugin("SofaEngine")
    SofaRuntime.importPlugin("SofaGeneralEngine")

    root = Sofa.Core.Node()
    root.dt = 0.01
    root.name = 'root'
    root.gravity = [0.0, 0.0, 0.0]
    createScene(root)

    Sofa.Simulation.init(root)
    Sofa.Gui.GUIManager.Init("Kromagnon scene", "qglviewer")
    Sofa.Gui.GUIManager.createGUI(root, __file__)
    Sofa.Gui.GUIManager.SetDimension(1080, 600)
    Sofa.Gui.GUIManager.MainLoop(root)
