# Basic python imports
import copy
import os
import numpy as np
import torch
import functools

# Sofa related imports
import SofaRuntime

# DeepPhysX's Core imports
from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX_Core.Pipelines.BaseTrainer import BaseTrainer
from DeepPhysX_Core.Visualizer.MeshVisualizer import MeshVisualizer

# DeepPhysX's Sofa imports
from DeepPhysX_Sofa.Environment.SofaEnvironment import SofaEnvironment
from DeepPhysX_Sofa.Environment.SofaEnvironmentConfig import SofaEnvironmentConfig

# DeepPhysX's Pytorch imports
from DeepPhysX_PyTorch.FC.FCConfig import FCConfig


# ENVIRONMENT PARAMETERS
grid_params = {'grid_resolution': [25, 5, 5],  # Number of slices along each axis
               'grid_min': [0., 0., 0.],  # Lowest point of the grid
               'grid_max': [100, 15, 15],  # Highest point of the grid
               'fixed_box': [0., 0., 0., 0., 15, 15]}  # Points withing this box will be fixed by Sofa

grid_node_count = functools.reduce(lambda a, b: a * b, grid_params['grid_resolution'])
grid_dofs_count = grid_node_count * 3

# TRAINING PARAMETERS
nb_hidden_layers = 2
layers_dim = [grid_dofs_count] * (nb_hidden_layers + 2)  # nb_hidden_layer + input_layer + output_layer
nb_epoch = 100
nb_batch = 15
batch_size = 32


# Inherit from SofaEnvironment which allow to implement and create a Sofa scene in the DeepPhysX_Core pipeline
class FEMBeam(SofaEnvironment):

    def __init__(self, root_node, config, idx_instance=1):
        super(FEMBeam, self).__init__(root_node, config, idx_instance)
        # Scene configuration
        self.config = config

        # Add the listed plugin to Sofa environment so we can run the scene
        SofaRuntime.PluginRepository.addFirstPath(os.environ['CARIBOU_INSTALL'])
        required_plugins = ['SofaComponentAll', 'SofaCaribou', 'SofaBaseTopology',
                            'SofaEngine', 'SofaBoundaryCondition', 'SofaTopologyMapping']
        root_node.addObject('RequiredPlugin', pluginName=required_plugins)

    def create(self):
        """
        Create the Sofa scene graph. Automatically called by SofaEnvironmentConfig.
        :return: None
        """
        #
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
        self.root.beamFEM.addObject('RegularGridTopology',
                                    name='Grid',
                                    min=grid_params['grid_min'],
                                    max=grid_params['grid_max'],
                                    nx=grid_params['grid_resolution'][0],
                                    ny=grid_params['grid_resolution'][1],
                                    nz=grid_params['grid_resolution'][2])

        self.MO = self.root.beamFEM.addObject('MechanicalObject', src='@Grid', name='MO', template='Vec3d',
                                              showObject=True)

        # Volume topology of the grid
        self.root.beamFEM.addObject('HexahedronSetTopologyContainer', name='Hexa_Topology', src='@Grid')
        self.root.beamFEM.addObject('HexahedronSetGeometryAlgorithms', template='Vec3d')
        self.root.beamFEM.addObject('HexahedronSetTopologyModifier')

        # Surface topology of the grid
        self.surface = self.root.beamFEM.addObject('QuadSetTopologyContainer', name='Quad_Topology',
                                                   src='@Hexa_Topology')
        self.root.beamFEM.addObject('QuadSetGeometryAlgorithms', template='Vec3d')
        self.root.beamFEM.addObject('QuadSetTopologyModifier')
        self.root.beamFEM.addObject('Hexa2QuadTopologicalMapping', input="@Hexa_Topology", output="@Quad_Topology")

        # Constitutive law of the beam
        self.root.beamFEM.addObject('NeoHookeanMaterial', young_modulus=4500, poisson_ratio=0.45, name="StVK")
        self.root.beamFEM.addObject('HyperelasticForcefield', material="@StVK", template="Hexahedron",
                                    topology='@Hexa_Topology', printLog=True)

        # Fixed section of the beam
        self.root.beamFEM.addObject('BoxROI', box=grid_params['fixed_box'], name='Fixed_Box')
        self.root.beamFEM.addObject('FixedConstraint', indices='@Fixed_Box.indices')

        # Forcefield through which the external forces are applied
        self.CFF = self.root.beamFEM.addObject('ConstantForceField', name='CFF', showArrowSize='0.1',
                                               forces=[0 for _ in range(3 * grid_node_count)],
                                               indices=list(iter(range(grid_node_count))))

    def onSimulationInitDoneEvent(self, event):
        """
        Called within the Sofa pipeline at the end of the scene graph initialisation.
        :param event: Sofa Event
        :return: None
        """
        # Get the data sizes
        self.input_size = self.MO.position.value.shape
        self.output_size = self.MO.position.value.shape
        visu = self.getDataManager().visualizer_manager.visualizer
        if self.environment_manager is not None:
            visu.addObject(positions=self.MO.position.value, cells=self.surface.quads.value)

    def onAnimateBeginEvent(self, event):
        """
        Called within the Sofa pipeline at the beginning of the time step.
        :param event: Sofa Event
        :return: None
        """
        # Reset position
        self.MO.position.value = self.MO.rest_position.value

        # Create a random force
        F = np.random.random(3) - np.random.random(3)
        K = np.random.randint(1, 3)
        F = (K / np.linalg.norm(F)) * F
        self.CFF.force.value = F

    def onAnimateEndEvent(self, event):
        """
        Called within the Sofa pipeline at the end of the time step.
        :param event: Sofa Event
        :return: None
        """
        # Render
        self.getDataManager().visualizer_manager.visualizer.render()

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
        return self.root.beamFEM.ODESolver.converged.value

    def applyPrediction(self, prediction):
        """
        Apply the prediction of the network in the Sofa environment. Automatically called by EnvironmentManager.
        :return: None
        """
        # Needed for prediction only
        pass

    def close(self):
        quit(0)


def createScene(root_node=None):
    # Environment config
    env_config = SofaEnvironmentConfig(environment_class=FEMBeam,
                                       root_node=root_node,
                                       always_create_data=False)

    # Network config
    net_config = FCConfig(network_name="beam_FC",
                          save_each_epoch=False,
                          loss=torch.nn.MSELoss,
                          lr=1e-5,
                          optimizer=torch.optim.Adam,
                          dim_output=3,
                          dim_layers=layers_dim)

    # Dataset config
    dataset_config = BaseDatasetConfig(partition_size=1,
                                       shuffle_dataset=True)

    trainer = BaseTrainer(session_name="trainings/Example_training",
                          visualizer_class=MeshVisualizer,
                          dataset_config=dataset_config,
                          environment_config=env_config,
                          network_config=net_config,
                          nb_epochs=nb_epoch,
                          nb_batches=nb_batch,
                          batch_size=batch_size)

    trainer.execute()


if __name__ == '__main__':
    createScene()
