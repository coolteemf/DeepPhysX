import os
import sys
import copy
import numpy as np
import random
from time import clock_gettime_ns as timer
import time

import Sofa.Gui
import SofaRuntime
import Sofa.Simulation
from Caribou.Topology import Grid3D
import Resources.Runner.UnetRunner as UR
import NeuralNetwork.unet.network as unet
import Resources.tensorTransformUtils as ttu
from Resources import sparseGridUtils as sgu
import Resources.metrics as metrics


# File paths
meshfile = "/home/robin/dev/kromagnon-master/src/Data/mesh_kromagnon/liver/patient3/patient3_parenchyme_aligned_and_trimmed_simplified.obj"
network_name = "patient3_test"
dir_pred = "predictions_patient3"
path_to_save = "../../../Prediction/Data/" + dir_pred

# Force parameters
amplitude_max = 0.05
nb_simultaneous_forces = 20
threshold_distance = 0.06
roi_radii = 0.015

# Transformation to center the liver
translation = [-0.0134716, 0.021525, -0.427]

# Fixed constraints for centered liver
fixed_point = np.array([-0.00338, -0.0256, 0.52]) + np.array(translation)
fixed_width = np.array([0.07, 0.05, 0.04])
fixed_box = (fixed_point - fixed_width / 2.).tolist() + (fixed_point + fixed_width / 2.).tolist()

# # Compute grid resolution for desired cell size and bbox size centered at (0, 0, 0)
# grid_size = 20
# bbox_size = 0.3
# minbb = [- bbox_size * 0.5, - bbox_size * 0.5, - bbox_size * 0.5]
# maxbb = [  bbox_size * 0.5,   bbox_size * 0.5,   bbox_size * 0.5]
# grid_resolution = [grid_size, grid_size, grid_size]
# print("Grid resolution is {}".format(grid_resolution))
# cell_size = 0.1
# minbb = np.array([-0.0773835, -0.0767571, 0.381479]) + np.array(translation)
# maxbb = np.array([0.107494, 0.141429, 0.467182]) + np.array(translation)
# grid_resolution = sgu.compute_grid_resolution(maxbb, minbb, cell_size)
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

# Validation metrics: compute TRE on randomly placed markers
rest_position_markers = np.array([
    [ 0.02000, -0.0000, 0.46],
    [ 0.01000, -0.0900, 0.50],
    [-0.06338, -0.0800, 0.49],
    [-0.01500,  0.0300, 0.47],
])


class Liver(UR.UnetRunner):
    def __init__(self, node, *args, **kwargs):
        UR.UnetRunner.__init__(self,
                               node=node,
                               directory=dir_pred,
                               dataScale=1000.0,
                               inputShape=[grid_resolution[2], grid_resolution[1], grid_resolution[0], 1],
                               outputShape=[grid_resolution[2], grid_resolution[1], grid_resolution[0], 3],
                               *args, **kwargs)
        self.forcefield = None
        self.solver = None
        self.sparsegrid = None
        self.behavior_state = None
        self.elastic_law = None
        self.intermediate_grid = None
        self.intermediate_behavior_state = None
        self.indices_of_sparsegrid_in_regulargrid = None
        self.indices_of_regulargrid_in_sparsegrid = None
        self.regular_grid_rest_shape_position = None
        self.geometry_mask = None
        self.area_to_compute_metrics = None
        self.indices_on_regular_grid_for_metrics = None
        self.behavior_state_pred = None
        self.behavior_state_gt = None
        self.visible_nodes_in_sparse_grid = None
        self.visible_nodes_regular_grid_rest_shape_position = None
        self.visible_nodes_in_regular_grid = None
        self.visible_surface_nodes_position_for_input_saving = None
        self.visible_surface_nodes_position_for_force_application = None
        self.nb_nodes_sparse_grid = None
        self.voxelization_times = []
        self.complete_prediction_times = []
        self.GPU_prediction_times = []


        self.nb_nodes_regular_grid = self.input_shape[0] * self.input_shape[1] * self.input_shape[2]
        self.totalPredCount = 0

        config = unet.UNetConfig(
            steps=3,
            first_layer_channels=128,
            num_classes=self.output_shape[3],
            num_input_channels=self.input_shape[3],
            ndims=3,
            border_mode='same',
            two_sublayers=True
        )
        self.manager.setNetwork(config,
                                network=unet.UNet,
                                network_dir="Training/Data/" + network_name + "/network")

        self.createGraph(node)

    def getInput(self):
        start = timer(time.CLOCK_REALTIME)
        # Select random amount of visible surface
        indices = [random.randint(0, len(self.rgbd_pcd_mo.position.array()) - 1) for i in range(random.randint(700, 800))]
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

        end = timer(time.CLOCK_REALTIME)
        self.voxelization_times.append((end - start) * 1e-6)
        # print("Voxelization using Caribou grids took {} milliseconds.".format((end-start)*1e-6))
        self.inputs = np.concatenate((self.inputs, DF_grid[None, :]))

    def getOutput(self):
        actual_positions_on_regular_grid = np.zeros((self.nb_nodes_regular_grid, 3), dtype=np.double)
        actual_positions_on_regular_grid[
            self.indices_of_sparsegrid_in_regulargrid] = self.behavior_state.position.array()
        self.outputs = np.concatenate(
            (self.outputs, copy.copy(np.subtract(actual_positions_on_regular_grid, self.regular_grid_rest_shape_position[None, :]))))
        self.rootNode.Loop.currentBatchSize = len(self.outputs)

    def onSimulationInitDoneEvent(self, value):
        # Initialize mapping between sparse grid and regular grid
        self.indices_of_sparsegrid_in_regulargrid,\
        self.indices_of_regulargrid_in_sparsegrid,\
        self.regular_grid_rest_shape_position = sgu.from_sparse_to_regular_grid(
            self.nb_nodes_regular_grid,
            self.sparsegrid,
            self.behavior_state)
        self.nb_nodes_sparse_grid = len(self.behavior_state_pred.rest_position.value)
        print("Number of nodes in sparse grid is {}".format(self.nb_nodes_sparse_grid))
        print("Number of nodes in regular grid is {}".format(self.nb_nodes_regular_grid))
        self.inputs = np.empty((0, *(self.nb_nodes_regular_grid, 1)))
        self.outputs = np.empty((0, *(self.nb_nodes_regular_grid, 3)))

    def onAnimateEndEvent(self, value):
        print("self.totalPredCount", self.totalPredCount)
        self.totalPredCount += 1
        # Get online input and output
        self.getInput()
        self.getOutput()

        # Predict
        start = timer(time.CLOCK_REALTIME)
        pred, gt = self.Predict()
        end = timer(time.CLOCK_REALTIME)
        print("Complete prediction with data loading to GPU takes {} milliseconds.".format((end-start)*1e-6))
        self.complete_prediction_times.append((end-start)*1e-6)

        # Unpadd and rescale back
        pred = ttu.inverse_pad(pred, [(0, 0)] + self.pad_widths) / self.dataScale
        gt = ttu.inverse_pad(gt, [(0, 0)] + self.pad_widths) / self.dataScale
        # Transform pred and gt
        pred = np.array(pred.cpu())
        pred = np.transpose(pred[0], (1, 2, 3, 0))
        gt = np.array(gt.cpu())
        gt = np.transpose(gt[0], (1, 2, 3, 0))

        # Update mechanical objects' position
        self.UpdateMechanicalObjectPosition(pred, gt)

        # Compute general metrics
        diff_reg_grid = gt - pred
        diff_reg_grid_3 = diff_reg_grid.reshape(self.nb_nodes_regular_grid, 3)
        diff_sparse_grid_3 = diff_reg_grid_3[self.indices_of_sparsegrid_in_regulargrid]
        self.manager.dataManager.add_customScalarFull('L2_norm_sparsegrid',
                                                      np.linalg.norm(diff_sparse_grid_3),
                                                      self.totalPredCount)
        self.manager.dataManager.add_customScalarFull('max_nodal_L2_norm_sparsegrid',
                                                      metrics.max_nodal_euclidean_norm(diff_sparse_grid_3),
                                                      self.totalPredCount)

        # Compute average, min and max TRE
        markers_gt_positions = np.array(self.markers_gt_state.position.value)
        markers_pred_positions = np.array(self.markers_pred_state.position.value)
        norms_at_markers = np.array([])
        for marker in range(len(markers_gt_positions)):
            norms_at_markers = np.concatenate((norms_at_markers,
                                               np.array([np.linalg.norm(markers_pred_positions[marker]
                                                                        - markers_gt_positions[marker])])))
        print("Average TRE over the {} markers = {}".format(len(markers_gt_positions), norms_at_markers.mean()))
        print("Std of  TRE over the {} markers = {}".format(len(markers_gt_positions), norms_at_markers.std()))
        print("Maximal TRE over the {} markers = {}".format(len(markers_gt_positions), norms_at_markers.max()))
        print("Minimal TRE over the {} markers = {}".format(len(markers_gt_positions), norms_at_markers.min()))
        self.manager.dataManager.add_customScalarFull('mean_TRE', norms_at_markers.mean(), self.totalPredCount)
        self.manager.dataManager.add_customScalarFull('std_TRE', norms_at_markers.std(), self.totalPredCount)
        self.manager.dataManager.add_customScalarFull('min_TRE', norms_at_markers.min(), self.totalPredCount)
        self.manager.dataManager.add_customScalarFull('max_TRE', norms_at_markers.max(), self.totalPredCount)

        print("Prediction times statistics excluding first sample")
        print("Voxelization time {} +- {} milliseconds".format(np.array(self.voxelization_times)[1:].mean(),
                                                          np.array(self.voxelization_times)[1:].std()))
        print("Complete prediction takes {} +- {} milliseconds".format(np.array(self.complete_prediction_times)[1:].mean(),
                                                          np.array(self.complete_prediction_times)[1:].std()))
        print("Pure GPU prediction {} +- {} milliseconds".format(np.array(self.GPU_prediction_times)[1:].mean(),
                                                          np.array(self.GPU_prediction_times)[1:].std()))

        # Generate next forces
        selected_centers = np.empty([0, 3])
        for j in range(nb_simultaneous_forces):
            distance_check = True
            f = np.random.uniform(low=-1, high=1, size=(3,))
            f = (f / np.linalg.norm(f)) * amplitude_max * np.random.random(1)

            # Pick up a random visible surface point and apply translation
            current_point = self.visible_surface_nodes_position[
                np.random.randint(len(self.visible_surface_nodes_position))]
            # current_point += np.array(translation)

            # Check if the current point is far enough from the already selected points
            for p in range(selected_centers.shape[0]):
                distance = np.linalg.norm(current_point - selected_centers[p])
                if distance < threshold_distance:
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
                self.spheres[j].centers.value = [[0.0, 0.0, 0.0]]
                self.force_fields[j].indices.value = [0]
                self.force_fields[j].forces.value = [[0.0, 0.0, 0.0]]

        self.eraseInPutAndOutPut()

    def createGraph(self, rootNode):
        surface_mesh_loader = rootNode.addObject('MeshObjLoader', name='surface_mesh', filename=meshfile,
                                                 translation=translation)
        self.visible_surface_nodes_position = sgu.extract_visible_mo(
            camera_position=np.array([-0.177458, 0.232606, 0.780813]),
            normals=surface_mesh_loader.normals.value,
            positions=surface_mesh_loader.position.value,
            dot_threshold=0.0, rand_threshold=0.0, distance_from_camera_threshold=1e6)

        # Grid3D from caribou in order to encode the RGB-D point cloud as a distance field
        self.grid_df = Grid3D(anchor_position=minbb, n=[nb_cells_x, nb_cells_y, nb_cells_z],
                              size=bbox_size)
        print("Number of nodes in grid:", self.grid_df.number_of_nodes())
        print("Number of cells in grid:", self.grid_df.number_of_cells())

        #  /root/mechanical_node/
        mechanical_node = rootNode.addChild('mechanical')

        # Time integration using caribou components
        self.solver = mechanical_node.addObject(
            'StaticODESolver',
            newton_iterations=10,
            correction_tolerance_threshold=1e-6,
            residual_tolerance_threshold=1e-6,
            shoud_diverge_when_residual_is_growing=False,
            printLog=True
        )
        mechanical_node.addObject(
            'ConjugateGradientSolver',
            name='CGSolver',
            preconditioning_method='Diagonal',
            maximum_number_of_iterations=2000,
            residual_tolerance_threshold=1.e-9,
            printLog=True
        )

        # Sparse grid topology
        self.sparsegrid = mechanical_node.addObject(
            'SparseGridTopology',
            name='grid',
            min=minbb,
            max=maxbb,
            n=[self.input_shape[2], self.input_shape[1], self.input_shape[0]],
            src='@../surface_mesh'
        )

        # Mechanical object
        self.behavior_state = mechanical_node.addObject(
            'MechanicalObject',
            src='@grid', name='mo', showObject=False
        )

        # Saint Venant Kirchoff material and Force Field
        mechanical_node.addObject(
            'SaintVenantKirchhoffMaterial',
            # 'NeoHookeanMaterial',
            young_modulus=5000,
            poisson_ratio=0.4,
            name="stvk"
        )
        mechanical_node.addObject(
            'HyperelasticForcefield',
            material="@stvk",
            template="Hexahedron", # Looks for an hexa topology
            printLog=True
        )
        # Fixed boundary conditions
        mechanical_node.addObject('BoxROI', box=fixed_box, name='fixed_box_roi', drawBoxes=True)
        mechanical_node.addObject('FixedConstraint', indices='@fixed_box_roi.indices')

        for i in range(len(rest_position_markers)):
            mechanical_node.addObject('SphereROI', name='sphere_marker' + str(i),
                                      centers=str(np.array(rest_position_markers[i])+np.array(translation))[1:-1],
                                      radii='0.01', drawSphere=False)

#  /root/mechanical_node/embedded_surface
        embedded_surface = mechanical_node.addChild('embedded_surface')
        embedded_surface.addObject('MechanicalObject', name='mo_embedded_surface', src='@../../surface_mesh')
        embedded_surface.addObject('TriangleSetTopologyContainer', name="triangleTopo", src="@../../surface_mesh")

        self.force_fields = []
        self.spheres = []
        for i in range(nb_simultaneous_forces):
            self.spheres.append(
                embedded_surface.addObject('SphereROI', name='sphere' + str(i),
                                           centers=str(fixed_point)[1:-1], radii=roi_radii, drawSphere=False))
            self.force_fields.append(
                embedded_surface.addObject('ConstantForceField', name='cff' + str(i), indices='0',
                                           forces=[0.0, 0.0, 0.0], showArrowSize='0'))
        embedded_surface.addObject('BarycentricMapping', input='@../mo', output='@./')

#  /root/mechanical_node/embedded_surface/visible_points
        visible_points = embedded_surface.addChild('visible_points')
        self.learnOn_mo = visible_points.addObject('MechanicalObject', tags="learnOn", name='mo_learnOn',
                                                   position=self.visible_surface_nodes_position,
                                                   showObject=False,
                                                   showObjectScale=5, showColor=[0, 1, 1, 1])#, translation=translation)

#  /root/mechanical_node/embedded_surface/rgbd_pcd
        rgbd_pcd = embedded_surface.addChild('rgbd_pcd')
        self.rgbd_pcd_mo = rgbd_pcd.addObject('MechanicalObject', name='mo_rgbd_pcd',
                                              position=self.visible_surface_nodes_position,
                                              showObject=False, showObjectScale=5, showColor=[1, 0, 0, 1])#,
                                              # translation=translation)
        rgbd_pcd.addObject('BarycentricMapping', input='@../../mo', output='@./')

#  /root/mechanical_node/network_prediction
        network_prediction = mechanical_node.addChild('network_prediction')
        self.behavior_state_pred = network_prediction.addObject('MechanicalObject', src='@../grid',
                                                                name='behavior_state_pred')
#  /root/mechanical_node/network_prediction/markers
        markers = network_prediction.addChild('markers')
        self.markers_pred_state = markers.addObject('MechanicalObject', position=rest_position_markers.tolist(),
                                                    showObject=False, showObjectScale=20, showColor=[0, 0.5, 0, 1])#,
                                                    # translation=translation)
        markers.addObject('BarycentricMapping', input='@../behavior_state_pred', output='@./')
#  /root/mechanical_node/network_prediction/visual
        visual = network_prediction.addChild('visual')
        # visual.addObject('VisualStyle', displayFlags='hideWireframe')
        visual.addObject('OglModel', src='@../../../surface_mesh', color='green')
        visual.addObject('BarycentricMapping', input='@../behavior_state_pred', output='@./')


#  /root/mechanical_node/sofa_gt
        sofa_gt = mechanical_node.addChild('sofa_gt')
        self.behavior_state_gt = sofa_gt.addObject('MechanicalObject', src='@../grid', name='behavior_state_gt')
#  /root/mechanical_node/sofa_gt/markers
        markers = sofa_gt.addChild('markers')
        self.markers_gt_state = markers.addObject('MechanicalObject', position=rest_position_markers.tolist(),
                                                  showObject=False, showObjectScale=20, showColor=[0.5, 0, 0, 1])#,
                                                  # translation=translation)
        markers.addObject('BarycentricMapping', input='@../behavior_state_gt', output='@./')
#  /root/mechanical_node/sofa_gt/visual
        visual = sofa_gt.addChild('visual')
        # visual.addObject('VisualStyle', displayFlags='hideWireframe')
        visual.addObject('OglModel', src='@../../../surface_mesh', color='red')
        visual.addObject('BarycentricMapping', input='@../behavior_state_gt', output='@./')


#  /root/mechanical_node/rest_shape
        rest_shape = mechanical_node.addChild('rest_shape')
        rest_shape.addObject('MechanicalObject', src='@../grid', name='mo_rest_shape')
        visual_rest_shape = rest_shape.addChild('visual_rest_shape')
        visual_rest_shape.addObject('OglModel', src='@../../../surface_mesh', color='gray')



def createScene(rootNode):
    rootNode.addObject('DefaultAnimationLoop', name="Loop")
    rootNode.addObject('AttachBodyButtonSetting', name='mouse', stiffness='1.0')
    rootNode.addObject('DefaultPipeline', depth='6', verbose='0', draw='0', name='DefaultCollisionPipeline')
    rootNode.addObject('DiscreteIntersection', name='Intersection')
    rootNode.addObject('ViewerSetting', name='viewer')
    # rootNode.addObject('BackgroundSetting', name='ViewerBGcolor', color='0.1 0.1 0.1')
    rootNode.addObject('VisualStyle', displayFlags='showVisualModels showBehaviorModels hideForceFields showWireframe')
    rootNode.addObject(Liver(rootNode))
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
