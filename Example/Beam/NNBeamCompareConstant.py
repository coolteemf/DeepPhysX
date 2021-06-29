"""
Prediction scene: NN simulated beam with random deformations compared to the FEM ground truth
(launch with the script beamPrediction in FC repository)
"""

import copy
import numpy as np
import random

from DeepPhysX_Sofa.Environment.SofaEnvironment import SofaEnvironment


class NNBeamCompareConstant(SofaEnvironment):

    def __init__(self, root_node, config, idx_instance=1, training=True):
        super(NNBeamCompareConstant, self).__init__(root_node, config, idx_instance)
        self.config = config
        self.nb_steps = 0

    def create(self, config):
        # Get the scene parameters
        p_grid = config.p_grid
        g_res = p_grid['grid_resolution']
        self.nb_node = g_res[0] * g_res[1] * g_res[2]
        self.grid_size = config.p_grid['min'] + config.p_grid['max']

        # Visual style of the scene
        self.root.addObject('VisualStyle', displayFlags="showVisualModels")

        # BEAM FEM
        self.root.addChild('beamFEM')
        # ODE solver + static solver
        self.root.beamFEM.addObject('LegacyStaticODESolver', name='StaticSolver', newton_iterations=50,
                                    correction_tolerance_threshold=1e-6, residual_tolerance_threshold=1e-6,
                                    printLog=False)
        self.root.beamFEM.addObject('ConjugateGradientSolver', name='LinearSolver', preconditioning_method='Diagonal',
                                    maximum_number_of_iterations=500, residual_tolerance_threshold=1e-9,
                                    printLog=False)
        # Grid topology of the beam
        self.root.beamFEM.addObject('RegularGridTopology', name='Grid', min=p_grid['min'], max=p_grid['max'],
                                    nx=g_res[0], ny=g_res[1], nz=g_res[2])
        self.femMO = self.root.beamFEM.addObject('MechanicalObject', src='@Grid', name='MO', showObject=False)
        self.root.beamFEM.addObject('HexahedronSetTopologyContainer', name='Hexa_Topology', src='@Grid')
        self.root.beamFEM.addObject('HexahedronSetGeometryAlgorithms', template='Vec3d')
        self.root.beamFEM.addObject('HexahedronSetTopologyModifier')
        # Surface of the grid
        self.femSurface = self.root.beamFEM.addObject('QuadSetTopologyContainer', name='Quad_Topology',
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
        # # External forces applied on the surface
        # self.femBox = self.root.beamFEM.addObject('BoxROI', name='ForceBox', box=self.grid_size, drawBoxes=True,
        #                                           drawSize=1)
        # self.femCFF = self.root.beamFEM.addObject('ConstantForceField', name='CFF', showArrowSize='0.1',
        #                                           forces=[0 for _ in range(3 * self.nb_node)],
        #                                           indices=list(iter(range(self.nb_node))))
        # Visual model
        self.root.beamFEM.addChild('Visual')
        self.root.beamFEM.Visual.addObject('OglModel', name="oglModel", src='@../Grid', color='white')
        self.root.beamFEM.Visual.addObject('BarycentricMapping', name="BaryMap2", input='@../MO', output='@./')

        # BEAM NN
        self.root.addChild('beamNN')
        # "Fake" solver (zero newton iteration, only to compute forces)
        self.root.beamFEM.addObject('LegacyStaticODESolver', name='StaticSolver', newton_iterations=0,
                                    correction_tolerance_threshold=1e-6, residual_tolerance_threshold=1e-6,
                                    printLog=False)
        # Grid topology of the beam
        grid_min = [p_grid['min'][0], p_grid['min'][1], p_grid['min'][2] - 2 * p_grid['max'][2]]
        grid_max = [p_grid['max'][0], p_grid['max'][1], - p_grid['max'][2]]
        self.root.beamNN.addObject('RegularGridTopology', name='Grid', min=grid_min, max=grid_max,
                                   nx=g_res[0], ny=g_res[1], nz=g_res[2])
        self.nnMO = self.root.beamNN.addObject('MechanicalObject', src='@Grid', name='MO', showObject=True)
        self.root.beamNN.addObject('HexahedronSetTopologyContainer', name='Hexa_Topology', src='@Grid')
        self.root.beamNN.addObject('HexahedronSetGeometryAlgorithms', template='Vec3d')
        self.root.beamNN.addObject('HexahedronSetTopologyModifier')
        # Surface of the grid
        self.nnSurface = self.root.beamNN.addObject('QuadSetTopologyContainer', name='Quad_Topology',
                                                    src='@Hexa_Topology')
        self.root.beamNN.addObject('QuadSetGeometryAlgorithms', template='Vec3d')
        self.root.beamNN.addObject('QuadSetTopologyModifier')
        self.root.beamNN.addObject('Hexa2QuadTopologicalMapping', input="@Hexa_Topology", output="@Quad_Topology")
        # # External forces applied on the surface
        # box = [0., 0., 0., 100., 25., 25.]
        # self.nnBox = self.root.beamNN.addObject('BoxROI', name='ForceBox', box=box, drawBoxes=True,
        #                                         drawSize=1)
        # self.nnCFF = self.root.beamNN.addObject('ConstantForceField', name='CFF', showArrowSize='0.',
        #                                         forces=[0 for _ in range(3 * self.nb_node)],
        #                                         indices=list(iter(range(self.nb_node))))
        # Visual model
        self.root.beamNN.addChild('Visual')
        self.root.beamNN.Visual.addObject('OglModel', name="oglModel", src='@../Grid', color='white')
        self.root.beamNN.Visual.addObject('BarycentricMapping', name="BaryMap2", input='@../MO', output='@./')

    def onSimulationInitDoneEvent(self, event):
        # Get the data sizes
        self.input_size = self.nnMO.position.value.shape
        self.output_size = self.nnMO.position.value.shape
        # Get the indices of node on the surface
        self.idx_surface = self.femSurface.quads.value.reshape(-1)


        # Caution : the intersection between box and surface needs to be non empty
        x_min, x_max = 80, 90               # x_grid in [0., 100.] + margin of the box
        y_min, y_max = 10, 15.5             # y_grid in [0., 15.] + margin of the box
        z_min, z_max = -0.5, 15.5           # z_grid in [0., 15.] + margin of the box
        F = np.array([1., -0.5, -0.75])     # F_i in [-1., 1.]

        # Set a new bounding box
        # self.root.beamFEM.removeObject(self.femBox)
        self.femBox = self.root.beamFEM.addObject('BoxROI', name='ForceBox', drawBoxes=True, drawSize=1,
                                                  box=[x_min, y_min, z_min, x_max, y_max, z_max])
        self.femBox.init()
        # Get the intersection with the surface
        indices = list(self.femBox.indices.value)
        indices = list(set(indices).intersection(set(self.idx_surface)))
        # Same box for NN simulated beam
        # self.root.beamNN.removeObject(self.nnBox)
        z_min -= 2 * self.config.p_grid['max'][2]
        z_max -= 2 * self.config.p_grid['max'][2]
        self.nnBox = self.root.beamNN.addObject('BoxROI', name='ForceBox', drawBoxes=True, drawSize=1,
                                                box=[x_min, y_min, z_min, x_max, y_max, z_max])
        self.nnBox.init()
        # Create a random constant force field for the nodes in the bbox
        F = (10 / np.linalg.norm(F)) * F
        # Set a new constant force field (variable number of indices)
        # self.root.beamFEM.removeObject(self.femCFF)
        self.femCFF = self.root.beamFEM.addObject('ConstantForceField', name='CFF', showArrowSize='1',
                                                  indices=indices, force=list(F))
        self.femCFF.init()
        # self.root.beamNN.removeObject(self.nnCFF)
        self.nnCFF = self.root.beamNN.addObject('ConstantForceField', name='CFF', showArrowSize='1',
                                                indices=indices, force=list(F))
        self.nnCFF.init()

    def onAnimateBeginEvent(self, event):
        # Reset position
        self.femMO.position.value = self.femMO.rest_position.value
        self.nnMO.position.value = self.nnMO.rest_position.value

    def onAnimateEndEvent(self, event):
        # Count the steps
        self.nb_steps += 1

    def computeInput(self):
        # Compute the input force to give to the network
        # self.input = copy.copy(self.femMO.force.value)
        f = copy.copy(self.nnCFF.forces.value)
        ind = copy.copy(self.nnCFF.indices.value)
        F = np.zeros((self.nb_node, 3))
        for i in range(len(f)):
            F[ind[i]] = f[i]
        self.input = F

    def computeOutput(self):
        # Compute the output deformation to compare with the prediction of the network
        self.output = copy.copy(self.femMO.position.value - self.femMO.rest_position.value)

    def applyPrediction(self, prediction):
        # Add the displacement to the initial position
        U = prediction[0]
        self.nnMO.position.value = self.nnMO.rest_position.array() + U
