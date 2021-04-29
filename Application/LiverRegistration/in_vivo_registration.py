# Python imports
import os
import numpy as np
import glob
from time import clock_gettime_ns as timer
import time
import copy
import torch

# Kromagnon imports
from Resources import sparseGridUtils as sgu
import Resources.Runner.UnetRunner as UR
import NeuralNetwork.unet.network as unet
import Resources.tensorTransformUtils as ttu

# Sofa imports
import Sofa
import Sofa.Gui
import SofaRuntime
from Caribou.Topology import Grid3D


# computer = "fix"
# computer = "laptop"
computer = "sperry"
if computer == "fix":
    disk_with_data = "/media/andrea/Gertrude"
elif computer == "laptop":
    disk_with_data = "/media/andrea/data"
elif computer == "sperry":
    disk_with_data = "/home/sperry/data"

prefix = "/home/robin/dev/kromagnon-master/src/"
meshfile = prefix + "Data/mesh_kromagnon/liver/patient3/patient3_parenchyme_aligned_and_trimmed_simplified.obj"
tumourfile = prefix + "Data/mesh_kromagnon/liver/patient3/patient3_tumeurs_aligned.obj"
veinecavefile = prefix + "Data/mesh_kromagnon/liver/patient3/patient3_veinecave_aligned_and_trimmed.obj"
veineportefile = prefix + "Data/mesh_kromagnon/liver/patient3/patient3_veineporte_aligned_and_trimmed.obj"
pcdfiles = sorted(glob.glob(disk_with_data + '/patient3/imagesPatient3Liver1_0/pcd/segmented/pcd'+('[0-9]'*6)+'.txt'), key=lambda filename:int(filename[-10:-4]))[10:]
rgbfiles = sorted(glob.glob(disk_with_data + '/patient3/imagesPatient3Liver1_0/images/RGB/img1'+('[0-9]'*6)+'.png'), key=lambda filename:int(filename[-10:-4]))[10:]
print("{} FILES".format(len(pcdfiles)))
print("{} IMAGES".format(len(rgbfiles)))
network_name = "training_patient3"
dir_pred = network_name
path_to_save = "../../../Prediction/Data/" + dir_pred

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
#
cell_size = 0.07
margins = np.array([0.02, 0.02, 0.02])
minbb = np.array([-0.130815, -0.107192, 0.00732511]) - margins
maxbb = np.array([0.0544588, 0.0967464, 0.15144]) + margins
grid_resolution = sgu.compute_grid_resolution(maxbb, minbb, cell_size)
print("Grid resolution is {}".format(grid_resolution))
bbox_size = maxbb - minbb
nb_cells_x = grid_resolution[0] - 1
nb_cells_y = grid_resolution[1] - 1
nb_cells_z = grid_resolution[2] - 1
minbb = minbb.tolist()
maxbb = maxbb.tolist()


class Liver(UR.UnetRunner):
    def __init__(self, node, *args, **kwargs):
        UR.UnetRunner.__init__(self,
                               node=node,
                               directory="in_vivo_AR_patient3",
                               dataScale=1000.0,
                               inputShape=[grid_resolution[2], grid_resolution[1], grid_resolution[0], 1],
                               outputShape=[grid_resolution[2], grid_resolution[1], grid_resolution[0], 3],
                               *args, **kwargs)

        # SOFA Objects (will be available after the graph initialization)
        self.sparsegrid = None  # Sparse grid topology
        self.behavior_state = None  # Mechanical object containing the solution of the DOFs (the sparse grid nodes)
        self.constant_forcefield = None  # Constant force field that will apply forces on the nodes of the sparse grid
        self.initial_positions_in_sparsegrid = None  # The initial undeformed position of the sparse grid nodes
        self.indices_of_sparsegrid_in_regulargrid = None  # Mapping that gives the index of a node in the regular (complete) grid from its index in the sparse grid
        self.indices_of_regulargrid_in_sparsegrid = None  # Mapping that gives the index of a node in the sparse grid from its index in the regular (complete) grid
        self.pcl_grid_node_indices = None  # Indices of the sparse grid's nodes that enclose the point cloud
        self.pcl_state = None  # Mechanical object containing the position of the point cloud
        self.target_pcd_state = None
        self.target_pcd_normal = None
        self.visible_surface_nodes_position = None

        self.nb_nodes_regular_grid = grid_resolution[0] * grid_resolution[1] * grid_resolution[2]
        self.current_cloud_index = 0
        self.voxelization_times = []
        self.complete_prediction_times = []
        self.GPU_prediction_times = []

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
                                network_dir="../../../Training/Data/" + network_name + "/network")
        self.createGraph(node)

    def getInput(self):
        start = timer(time.CLOCK_REALTIME)
        # Initialize distance field to 0
        DF_grid = np.zeros((self.nb_nodes_regular_grid, 1), dtype=np.double)

        # Get list of nodes of the cells containing a point from the RGBD point cloud and
        # for each node, compute the distance to the considered point
        for p in self.rgbd_pcd_array:
            for node in self.grid_df.node_indices_of(self.grid_df.cell_index_containing(p)):
                if node < self.nb_nodes_regular_grid and DF_grid[node][0] == 0:
                    DF_grid[node] = np.linalg.norm(self.grid_df.node(node) - p)

        end = timer(time.CLOCK_REALTIME)
        print("Voxelization using Caribou grids took {} milliseconds.".format((end-start)*1e-6))

        self.inputs = np.concatenate((self.inputs, DF_grid[None, :]))

    def getOutput(self):
        # Can be anything. Useless in this prediction part
        self.outputs = np.concatenate(
            (self.outputs, copy.copy(np.zeros((self.nb_nodes_regular_grid, 3), dtype=np.double))))
        self.outputs.append()
        self.rootNode.Loop.currentBatchSize = len(self.outputs)

    def onSimulationInitDoneEvent(self, value):
        # Initialize mapping between sparse grid and regular grid
        self.indices_of_sparsegrid_in_regulargrid,\
        self.indices_of_regulargrid_in_sparsegrid,\
        self.regular_grid_rest_shape_position = sgu.from_sparse_to_regular_grid(
            self.nb_nodes_regular_grid,
            self.sparsegrid,
            self.behavior_state_pred)
        self.nb_nodes_sparse_grid = len(self.behavior_state_pred.rest_position.value)
        print("Number of nodes in sparse grid is {}".format(self.nb_nodes_sparse_grid))
        print("Number of nodes in regular grid is {}".format(self.nb_nodes_regular_grid))
        self.inputs = np.empty((0, *(self.nb_nodes_regular_grid, 1)))
        self.outputs = np.empty((0, *(self.nb_nodes_regular_grid, 3)))

    def createGraph(self, root):
        # Grid3D from caribou in order to encode the RGB-D point cloud as a distance field
        self.grid_df = Grid3D(anchor_position=minbb, n=[nb_cells_x, nb_cells_y, nb_cells_z], size=bbox_size)
        print("Number of nodes in grid:", self.grid_df.number_of_nodes())
        print("Number of cells in grid:", self.grid_df.number_of_cells())

#  /root
        surface_mesh = root.addObject(
            'MeshObjLoader',
            name='surface_mesh',
            filename=meshfile,
            translation=translation,
        )
        tumour_mesh = root.addObject(
            'MeshObjLoader',
            name='tumour_mesh',
            filename=tumourfile,
            translation=translation,
        )
        veinecave_mesh = root.addObject(
            'MeshObjLoader',
            name='veinecave_mesh',
            filename=veinecavefile,
            translation=translation,
        )
        veineporte_mesh = root.addObject(
            'MeshObjLoader',
            name='veineporte_mesh',
            filename=veineportefile,
            translation=translation,
        )

#  /root/target For visualization only
        target_node = root.addChild('target')
        self.target_pcd_state = target_node.addObject('MechanicalObject',
                                                      name="mo",
                                                      position=np.loadtxt(pcdfiles[0]).reshape((-1, 3)).tolist(),
                                                      showObject=True, showObjectScale=2, showColor=[1, 0, 0, 1])
#  /root/mechanical
        mechanical_node = root.addChild('mechanical')
        # Sparse grid
        self.sparsegrid = mechanical_node.addObject(
            'SparseGridTopology',
            name='sparsegrid',
            min=minbb,
            max=maxbb,
            n=[grid_resolution[0], grid_resolution[1], grid_resolution[2]],
            src='@../surface_mesh',
        )
        # Mechanical object
        self.behavior_state_pred = mechanical_node.addObject('MechanicalObject', src='@sparsegrid',
                                                             name='behavior_state', showObject=False)

#  /root/mechanical_node/visual_parenchyme
        visual_node = mechanical_node.addChild('visual_parenchyme')
        self.visual_parenchyme_state = visual_node.addObject('OglModel', src=surface_mesh.getLinkPath(),
                                                             color=(np.array([9., 244, 66, 50])/255.).tolist())
        visual_node.addObject('BarycentricMapping')

#  /root/mechanical_node/visual_tumour
        visual_node2 = mechanical_node.addChild('visual_tumour')
        self.visual_tumour_state = visual_node2.addObject('OglModel', src=tumour_mesh.getLinkPath(),
                                                          color=(np.array([244., 0, 0, 255]) / 255.).tolist())
        visual_node2.addObject('BarycentricMapping')

#  /root/mechanical_node/visual_veinecave
        visual_node3 = mechanical_node.addChild('visual_veinecave')
        self.visual_veinecave_state = visual_node3.addObject('OglModel', src=veinecave_mesh.getLinkPath(),
                                                             color=(np.array([0., 0, 244, 255]) / 255.).tolist())
        visual_node3.addObject('BarycentricMapping')

#  /root/mechanical_node/visual_veinecave
        visual_node4 = mechanical_node.addChild('visual_veineporte')
        self.visual_veineporte_state = visual_node4.addObject('OglModel', src=veineporte_mesh.getLinkPath(),
                                                              color=(np.array([0., 0, 244, 255]) / 255.).tolist())
        visual_node4.addObject('BarycentricMapping')

    def load_next_pcd(self):
        self.current_cloud_index += 1
        if self.current_cloud_index < len(pcdfiles):
            start = timer(time.CLOCK_REALTIME)
            print("LOADING POINT CLOUD '{}'".format(pcdfiles[self.current_cloud_index]))
            self.rgbd_pcd_array = np.loadtxt(pcdfiles[self.current_cloud_index]).reshape((-1, 3))
            self.rgbd_pcd_array += np.array(translation)
            self.target_pcd_state.position = self.rgbd_pcd_array.tolist()
            end = timer(time.CLOCK_REALTIME)
            print("PCD loading took {} milliseconds.".format((end - start) * 1e-6))

        if self.current_cloud_index < len(rgbfiles):
            start = timer(time.CLOCK_REALTIME)
            print("LOADING IMAGE '{}'".format(rgbfiles[self.current_cloud_index]))
            current_gui = Sofa.Gui.GUIManager.GetGUI()
            current_gui.setBackgroundImage(rgbfiles[self.current_cloud_index])
            end = timer(time.CLOCK_REALTIME)
            print("Background display took {} milliseconds.".format((end - start) * 1e-6))

    def onAnimateEndEvent(self, event):
        if self.current_cloud_index >= len(pcdfiles):
            return
        self.load_next_pcd()

        # Get online input and output
        self.getInput()
        self.getOutput()

        # Predict
        torch.cuda.synchronize()
        start = timer(time.CLOCK_REALTIME)
        pred, to_ignore = self.Predict()
        torch.cuda.synchronize()
        end = timer(time.CLOCK_REALTIME)
        print("Complete prediction with data loading to GPU takes {} milliseconds.".format((end - start) * 1e-6))
        self.complete_prediction_times.append((end - start) * 1e-6)

        # Unpadd and rescale back
        pred = ttu.inverse_pad(pred, [(0, 0)] + self.pad_widths) / self.dataScale

        # Transform pred
        torch.cuda.synchronize()
        start = timer(time.CLOCK_REALTIME)
        pred = np.array(pred.cpu())
        torch.cuda.synchronize()
        end = timer(time.CLOCK_REALTIME)
        print("Convert to cpu took {} milliseconds.".format((end - start) * 1e-6))
        pred = np.transpose(pred[0], (1, 2, 3, 0))

        # Update mechanical objects' position
        self.UpdatePosition(pred)

        self.eraseInPutAndOutPut()


def createScene(root):
    root.addObject('DefaultAnimationLoop', name="Loop")
    root.addObject(Liver(root))
    return root


if __name__ == "__main__":
    # # Register all the common component in the factory.
    SofaRuntime.importPlugin("SofaComponentAll")
    SofaRuntime.importPlugin("SofaOpenglVisual")
    SofaRuntime.importPlugin("SofaPython3")
    SofaRuntime.importPlugin("SofaCaribou")
    SofaRuntime.importPlugin("SofaKromagnon")
    SofaRuntime.importPlugin("CImgPlugin")



    root = Sofa.Core.Node()
    root.dt = 0.01
    root.name = 'root'
    root.gravity = [0.0, 0.0, 0.0]
    createScene(root)

    Sofa.Simulation.init(root)
    Sofa.Gui.GUIManager.Init("Kromagnon scene", "qglviewer")
    Sofa.Gui.GUIManager.createGUI(root, __file__)
    Sofa.Gui.GUIManager.SetDimension(640, 480)
    current_gui = Sofa.Gui.GUIManager.GetGUI()
    current_gui.setBackgroundImage(rgbfiles[0])
    Sofa.Gui.GUIManager.MainLoop(root)


