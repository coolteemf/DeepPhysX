"""
FEMBeam.py
FEM simulated beam with random deformations.
Can be launched as a Sofa scene using the 'runSofa.py' script in this repository.
Also used to train neural network in DeepPhysX_Core pipeline with the '../beamTrainingFC.py' script.
"""

import copy
import random
import numpy as np

from DeepPhysX_Sofa.Environment.SofaEnvironment import SofaEnvironment


# Inherit from SofaEnvironment which allow to implement and create a Sofa scene in the DeepPhysX_Core pipeline
class FEMBeam(SofaEnvironment):

    def __init__(self, root_node, config, idx_instance=1):
        super(FEMBeam, self).__init__(root_node, config, idx_instance)
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
        # Get the grid parameters (size, resolution)
        p_grid = config.p_grid
        g_res = p_grid['grid_resolution']
        self.nb_node = g_res[0] * g_res[1] * g_res[2]
        self.grid_size = config.p_grid['grid_min'] + config.p_grid['grid_max']

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
        self.root.beamFEM.Visual.addObject('OglModel', name="oglModel", src='@../Grid', color='white')
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

    def onAnimateBeginEvent(self, event):
        """
        Called within the Sofa pipeline at the beginning of the time step.
        :param event: Sofa Event
        :return: None
        """
        # Reset position
        self.MO.position.value = self.MO.rest_position.value
        # Create a random box ROI, select nodes of the surface
        indices = []
        while len(indices) == 0:    # We need the intersection between box and surface to be non empty
            x_min = random.randint(self.grid_size[0], self.grid_size[3] - 10)
            x_max = random.randint(x_min + 10, self.grid_size[3])
            y_min = random.randint(self.grid_size[1], self.grid_size[4] - 10)
            y_max = random.randint(y_min + 10, self.grid_size[4])
            z_min = random.randint(self.grid_size[2], self.grid_size[5] - 10)
            z_max = random.randint(z_min + 10, self.grid_size[5])
            # Set a new bounding box
            self.root.beamFEM.removeObject(self.box)
            self.box = self.root.beamFEM.addObject('BoxROI', name='ForceBox', drawBoxes=True, drawSize=1,
                                                   box=[x_min, y_min, z_min, x_max, y_max, z_max])
            self.box.init()
            # Get the intersection with the surface
            indices = list(self.box.indices.value)
            indices = list(set(indices).intersection(set(self.idx_surface)))
        # Create a random constant force field for the nodes in the bbox
        F = np.random.random(3) - np.random.random(3)
        K = np.random.randint(10, 25)
        F = (K / np.linalg.norm(F)) * F
        # Set a new constant force field (variable number of indices)
        self.root.beamFEM.removeObject(self.CFF)
        self.CFF = self.root.beamFEM.addObject('ConstantForceField', name='CFF', showArrowSize='0.1',
                                               indices=indices, force=list(F))
        self.CFF.init()

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

    def computeInput(self):
        """
        Compute the input to be given to the network. Automatically called by EnvironmentManager.
        :return: None
        """
        # Compute the input force to give to the network
        f = copy.copy(self.CFF.forces.value)
        ind = copy.copy(self.CFF.indices.value)
        F = np.zeros((self.nb_node, 3))
        # All forces are zero except on the CFF (intersection between box ROI and surface)
        for i in range(len(f)):
            F[ind[i]] = f[i]
        self.input = F

    def computeOutput(self):
        """
        Compute the output to be given to the network. Automatically called by EnvironmentManager.
        :return: None
        """
        # Compute the output deformation to compare with the prediction of the network
        self.output = copy.copy(self.MO.position.value - self.MO.rest_position.value)

    def applyPrediction(self, prediction):
        """
        Apply the prediction of the network in the Sofa environment. Automatically called by EnvironmentManager.
        :return: None
        """
        # Needed for prediction only
        pass
