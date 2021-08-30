# Basic python imports
import os
import numpy as np
import functools

# Sofa related imports
import SofaRuntime

# DeepPhysX's Core imports
from DeepPhysX_Core.Visualizer.MeshVisualizer import MeshVisualizer
from DeepPhysX_Core.Manager.EnvironmentManager import EnvironmentManager

# DeepPhysX's Sofa imports
from DeepPhysX_Sofa.Environment.SofaEnvironment import SofaEnvironment
from DeepPhysX_Sofa.Environment.SofaEnvironmentConfig import SofaEnvironmentConfig

# ENVIRONMENT PARAMETERS
grid_params = {'grid_resolution': [25, 5, 5],  # Number of slices along each axis
               'grid_min': [0., 0., 0.],  # Lowest point of the grid
               'grid_max': [100, 15, 15],  # Highest point of the grid
               'fixed_box': [0., 0., 0., 0., 15, 15]}  # Points withing this box will be fixed by Sofa

grid_node_count = functools.reduce(lambda a, b: a * b, grid_params['grid_resolution'])

vedo_visualizer = MeshVisualizer()


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
        self.root.beamFEM.addObject('FixedConstraint', indices='@Fixed_Box.indices', src='@MO')

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
        vedo_visualizer.addObject(positions=self.MO.position.value, cells=self.surface.quads.value)

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
        vedo_visualizer.render()


def createScene(root_node=None):
    # Environment config
    sofa_config = SofaEnvironmentConfig(environment_class=FEMBeam,
                                        root_node=root_node,
                                        always_create_data=False)

    env_manager = EnvironmentManager(environment_config=sofa_config)

    return env_manager.environment


if __name__ == '__main__':
    env = createScene()
    while True:
        env.step()
