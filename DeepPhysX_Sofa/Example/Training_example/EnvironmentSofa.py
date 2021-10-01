# Basic python imports
import os
import numpy as np
import functools

# DeepPhysX's Sofa imports
from DeepPhysX_Sofa.Environment.SofaEnvironment import SofaEnvironment

# Sofa related imports
import SofaRuntime

# Add the listed plugin to Sofa environment so we can run the scene
SofaRuntime.PluginRepository.addFirstPath(os.environ['CARIBOU_INSTALL'])
SofaRuntime.PluginRepository.addFirstPath(os.environ['DEEPPHYSICSSOFA_INSTALL'])
required_plugins = ['SofaComponentAll', 'SofaCaribou', 'DeepPhysicsSofa', 'SofaBaseTopology',
                    'SofaEngine', 'SofaBoundaryCondition', 'SofaTopologyMapping']

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

    def __init__(self, root_node, ip_address='localhost', port=10000, data_converter=None,
                 instance_id=1, number_of_instances=1):
        super(FEMBeam, self).__init__(ip_address=ip_address, port=port, data_converter=data_converter,
                                      instance_id=instance_id, number_of_instances=1, root_node=root_node)
        root_node.addObject('RequiredPlugin', pluginName=required_plugins)

    def create(self):
        """
        Create the Sofa scene graph. Automatically called by SofaEnvironmentConfig.
        :return: None
        """
        print(f"Created Env n°{self.instance_id}")
        #
        # BEAM FEM NODE
        self.root.addChild('beamFEM')
        # # ODE solver + static solver
        # self.root.beamFEM.addObject('LegacyStaticODESolver', name='ODESolver', newton_iterations=20,
        #                             correction_tolerance_threshold=1e-6, residual_tolerance_threshold=1e-6,
        #                             printLog=False)
        # self.root.beamFEM.addObject('ConjugateGradientSolver', name='StaticSolver', preconditioning_method='Diagonal',
        #                             maximum_number_of_iterations=1000, residual_tolerance_threshold=1e-9,
        #                             printLog=False)
        # ODE solver + static solver
        self.root.beamFEM.addObject('HybridNewtonRaphson', name='ODESolver', solve_with_inversion=True, newton_iterations=20,
                                    correction_tolerance_threshold=1e-6, residual_tolerance_threshold=1e-6,
                                    printLog=False)
        self.root.beamFEM.addObject('LDLTSolver', name='StaticSolver', #preconditioning_method='Diagonal',
                                    #maximum_number_of_iterations=1000, residual_tolerance_threshold=1e-9,
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

    def send_visualization(self):
        # return self.visualizer.createObjectData(data_dict={}, positions=self.MO.position.value,
        #                                         cells=self.surface.quads.value, at=0)
        return {}

    # def update_visualization(self):
    #     visu_dict = self.visualizer.updateObjectData(data_dict={}, positions=self.MO.position.value)
    #     self.sync_send_visualization_data(visu_dict)

    def onAnimateBeginEvent(self, event):
        """
        Called within the Sofa pipeline at the beginning of the time step.
        :param event: Sofa Event
        :return: None
        """
        # Reset position
        self.MO.position.value = self.MO.rest_position.value
        print(f"{np.linalg.norm((self.MO.position.value).reshape(-1))=}")
        print(f"{np.linalg.norm((self.MO.rest_position.value).reshape(-1))=}")
        print(f"{np.linalg.norm((self.MO.position.value-self.MO.rest_position.value).reshape(-1))=}")

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
        if self.compute_essential_data:
            self.sync_send_training_data(network_input=self.CFF.forces.value,
                                         network_output=self.MO.position.value - self.MO.rest_position.value)
           # self.update_visualization()
        self.sync_send_command_done()

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
        print(f"Closing Env n°{self.instance_id}")

    def __str__(self):
        return f"Environment n°{self.instance_id} with tensor {self.tensor}"
