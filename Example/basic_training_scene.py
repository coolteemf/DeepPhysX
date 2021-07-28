"""
beamTrainingFC.py
Script used to train a FC network on FEM beam deformations
"""
"""
FEMBeam.py
FEM simulated beam with random deformations.
Can be launched as a Sofa scene using the 'runSofa.py' script in this repository.
Also used to train neural network in DeepPhysX_Core pipeline with the '../beamTrainingFC.py' script.
"""

import torch


from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX_Core.Pipelines.BaseTrainer import BaseTrainer
from DeepPhysX_Core.Visualizer.MeshVisualizer import MeshVisualizer


from DeepPhysX_PyTorch.FC.FCConfig import FCConfig

import copy
import os
import random
import numpy as np
import SofaRuntime

from DeepPhysX_Sofa.Environment.SofaEnvironment import SofaEnvironment
from DeepPhysX_Sofa.Environment.SofaEnvironmentConfig import SofaEnvironmentConfig

# ENVIRONMENT PARAMETERS
grid_resolution = [25, 5, 5]
grid_min = [0., 0., 0.]
grid_max = [100, 15, 15]
fixed_box = [0., 0., 0., 0., 15, 15]
p_grid = {'grid_resolution': grid_resolution, 'grid_min': grid_min, 'grid_max': grid_max, 'fixed_box': fixed_box}

# TRAINING PARAMETERS
nb_hidden_layers = 2
nb_node = grid_resolution[0] * grid_resolution[1] * grid_resolution[2]
layers_dim = [nb_node * 3] + [nb_node * 3 for _ in range(nb_hidden_layers + 1)] + [nb_node * 3]
nb_epoch = 100
nb_batch = 15
batch_size = 32


# Inherit from SofaEnvironment which allow to implement and create a Sofa scene in the DeepPhysX_Core pipeline
class FEMBeam(SofaEnvironment):

    def __init__(self, root_node, config, idx_instance=1, visualizer_class=None):
        super(FEMBeam, self).__init__(root_node, config, idx_instance, visualizer_class)
        # Scene configuration
        self.config = config
        # Keep a track of the actual step number and how many samples diverged during the animation
        self.nb_steps = 0
        self.nb_converged = 0.
        self.converged = False
        SofaRuntime.PluginRepository.addFirstPath(os.environ['CARIBOU_INSTALL'])
        required_plugins = ['SofaComponentAll', 'SofaLoader', 'SofaCaribou', 'SofaBaseTopology', 'SofaGeneralEngine',
                        'SofaEngine', 'SofaOpenglVisual', 'SofaBoundaryCondition', 'SofaTopologyMapping',
                        'SofaConstraint', 'SofaDeformable', 'SofaGeneralObjectInteraction', 'SofaBaseMechanics',
                        'SofaMiscCollision']
        root_node.addObject('RequiredPlugin', pluginName=required_plugins)

    def create(self):
        """
        Create the Sofa scene graph. Automatically called by SofaEnvironmentConfig.
        :return: None
        """
        # Get the grid parameters (size, resolution)
        g_res = p_grid['grid_resolution']
        self.nb_node = g_res[0] * g_res[1] * g_res[2]
        self.grid_size = p_grid['grid_min'] + p_grid['grid_max']

        # BEAM FEM NODE
        self.root.addChild('beamFEM')
        # ODE solver + static solver
        self.root.beamFEM.addObject('LegacyStaticODESolver', name='ODESolver', newton_iterations=20,
                                    correction_tolerance_threshold=1e-6, residual_tolerance_threshold=1e-6,
                                    printLog=False)
        self.root.beamFEM.addObject('ConjugateGradientSolver', name='StaticSolver', preconditioning_method='Diagonal',
                                    maximum_number_of_iterations=1000, residual_tolerance_threshold=1e-9,
                                    printLog=False)
        # Grid topology of the beam
        self.root.beamFEM.addObject('RegularGridTopology', name='Grid', min=p_grid['grid_min'], max=p_grid['grid_max'],
                                    nx=g_res[0], ny=g_res[1], nz=g_res[2])
        self.MO = self.root.beamFEM.addObject('MechanicalObject', src='@Grid', name='MO', template='Vec3d',
                                              showObject=True)
        self.root.beamFEM.addObject('HexahedronSetTopologyContainer', name='Hexa_Topology', src='@Grid')
        self.root.beamFEM.addObject('HexahedronSetGeometryAlgorithms', template='Vec3d')
        self.root.beamFEM.addObject('HexahedronSetTopologyModifier')
        # Surface of the grid
        self.surface = self.root.beamFEM.addObject('QuadSetTopologyContainer', name='Quad_Topology',
                                                   src='@Hexa_Topology')
        self.root.beamFEM.addObject('QuadSetGeometryAlgorithms', template='Vec3d')
        self.root.beamFEM.addObject('QuadSetTopologyModifier')
        self.root.beamFEM.addObject('Hexa2QuadTopologicalMapping', input="@Hexa_Topology", output="@Quad_Topology")
        # Simulated hyperelastic material
        self.root.beamFEM.addObject('NeoHookeanMaterial', young_modulus=4500, poisson_ratio=0.45, name="StVK")
        self.root.beamFEM.addObject('HyperelasticForcefield', material="@StVK", template="Hexahedron",
                                    topology='@Hexa_Topology', printLog=True)
        # Fixed section of the beam
        self.root.beamFEM.addObject('BoxROI', box=p_grid['fixed_box'], name='Fixed_Box')
        self.root.beamFEM.addObject('FixedConstraint', indices='@Fixed_Box.indices')
        # External forces applied on the surface
        self.box = self.root.beamFEM.addObject('BoxROI', name='ForceBox', box=self.grid_size, drawBoxes=True,
                                               drawSize=1)
        self.CFF = self.root.beamFEM.addObject('ConstantForceField', name='CFF', showArrowSize='0.1',
                                               forces=[0 for _ in range(3 * self.nb_node)],
                                               indices=list(iter(range(self.nb_node))))
        # Visual model
        self.root.beamFEM.addChild('Visual')
        self.boj = self.root.beamFEM.Visual.addObject('OglModel', name="oglModel", src='@../Grid', color='white')
        self.root.beamFEM.Visual.addObject('BarycentricMapping', name="BaryMap2", input='@../MO', output='@./')

    def onSimulationInitDoneEvent(self, event):
        """
        Called within the Sofa pipeline at the end of the scene graph initialisation.
        :param event: Sofa Event
        :return: None
        """
        # Get the data sizes
        self.input_size = self.MO.position.value.shape
        self.output_size = self.MO.position.value.shape
        # Get the indices of node on the surface
        self.idx_surface = self.surface.quads.value.reshape(-1)
        self.initVisualizer()

    def initVisualizer(self):
        # Visualizer
        if self.visualizer is not None:
            self.visualizer.addObject(positions=self.boj.position.value, at=0)
            self.visualizer.addObject(positions=self.boj.position.value, cells=self.surface.quads.value, at=0)

    def onAnimateBeginEvent(self, event):
        """
        Called within the Sofa pipeline at the beginning of the time step.
        :param event: Sofa Event
        :return: None
        """
        # Reset position
        self.MO.position.value = self.MO.rest_position.value

        # Create a random constant force field for the nodes in the bbox
        F = np.random.random(3) - np.random.random(3)
        K = np.random.randint(1, 3)
        F = (K / np.linalg.norm(F)) * F
        # Set a new constant force field (variable number of indices)
        self.CFF.force.value = F

    def onAnimateEndEvent(self, event):
        """
        Called within the Sofa pipeline at the end of the time step.
        :param event: Sofa Event
        :return: None
        """
        # Count the steps
        self.nb_steps += 1
        # Check whether if the solver diverged or not
        self.converged = self.root.beamFEM.ODESolver.converged.value
        self.nb_converged += self.root.beamFEM.ODESolver.converged.value
        if self.nb_steps % 50 == 0:
            print("Converge rate:", self.nb_converged / self.nb_steps)
        # Render
        self.renderVisualizer()

    def computeInput(self):
        """
        Compute the input to be given to the network. Automatically called by EnvironmentManager.
        :return: None
        """
        # Compute the input force to give to the network
        self.input = copy.copy(self.CFF.forces.value)

    def computeOutput(self):
        """
        Compute the output to be given to the network. Automatically called by EnvironmentManager.
        :return: None
        """
        # Compute the output deformation to compare with the prediction of the network
        self.output = copy.copy(self.MO.position.value - self.MO.rest_position.value)

    def checkSample(self, check_input=True, check_output=True):
        return self.converged

    def applyPrediction(self, prediction):
        """
        Apply the prediction of the network in the Sofa environment. Automatically called by EnvironmentManager.
        :return: None
        """
        # Needed for prediction only
        pass


def createScene(root_node=None):
    # Environment config
    env_config = SofaEnvironmentConfig(environment_class=FEMBeam, root_node=root_node, always_create_data=False,
                            visualizer_class=MeshVisualizer)

    # # Network config
    # net_config = FCConfig(network_name="beam_FC", save_each_epoch=False,
    #                       loss=torch.nn.MSELoss, lr=1e-5, optimizer=torch.optim.Adam,
    #                       dim_output=3, dim_layers=layers_dim)
    #
    # # Dataset config
    # dataset_config = BaseDatasetConfig(partition_size=1, shuffle_dataset=True)
    #
    # trainer = BaseTrainer(session_name="trainings/session_625", dataset_config=dataset_config,
    #                       environment_config=env_config, network_config=net_config,
    #                       nb_epochs=nb_epoch, nb_batches=nb_batch, batch_size=batch_size)

    # Manually create and init the environment from the configuration object
    env = env_config.createEnvironment()
    env_config.initSofaSimulation()
    env_config.addVisualizer(env)
    env.initVisualizer()
    return env.root


if __name__ == '__main__':
    root = createScene()
    import Sofa.Gui
    # Launch the GUI
    Sofa.Gui.GUIManager.Init("main", "qglviewer")
    Sofa.Gui.GUIManager.createGUI(root, __file__)
    Sofa.Gui.GUIManager.SetDimension(1080, 1080)
    Sofa.Gui.GUIManager.MainLoop(root)
    Sofa.Gui.GUIManager.closeGUI()
