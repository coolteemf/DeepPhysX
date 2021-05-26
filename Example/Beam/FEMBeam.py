import copy
import numpy as np

from DeepPhysX_Sofa.Environment.SofaBaseEnvironment import SofaBaseEnvironment


class FEMBeam(SofaBaseEnvironment):

    def __init__(self, root_node, config, idx_instance=1, training=True):
        super(FEMBeam, self).__init__(root_node, config, idx_instance)
        self.config = config

    def create(self, config):
        # Get parameters
        p_grid = config.p_grid
        g_res = p_grid['grid_resolution']
        self.nb_dof = g_res[0] * g_res[1] * g_res[2]

        # Beam FEM
        self.root.addChild('BeamFEM')
        self.root.BeamFEM.addObject('LegacyStaticODESolver', name='StaticSolver', newton_iterations=50,
                                    correction_tolerance_threshold=1e-6, residual_tolerance_threshold=1e-6,
                                    printLog=False)
        self.root.BeamFEM.addObject('ConjugateGradientSolver', name='LinearSolver', preconditioning_method='Diagonal',
                                    maximum_number_of_iterations=2000, residual_tolerance_threshold=1e-9,
                                    printLog=False)
        self.root.BeamFEM.addObject('RegularGridTopology', name='Grid', min=p_grid['min'], max=p_grid['max'],
                                    nx=g_res[0], ny=g_res[1], nz=g_res[2])
        self.root.BeamFEM.addObject('HexahedronSetTopologyContainer', name='Hexa_Topology', src='@Grid')
        self.root.BeamFEM.addObject('HexahedronSetGeometryAlgorithms', template='Vec3d')
        self.root.BeamFEM.addObject('HexahedronSetTopologyModifier')
        self.root.BeamFEM.addObject('QuadSetTopologyContainer', name='Quad_Topology', src='@Hexa_Topology')
        self.root.BeamFEM.addObject('QuadSetGeometryAlgorithms', template='Vec3d')
        self.root.BeamFEM.addObject('QuadSetTopologyModifier')
        self.root.BeamFEM.addObject('Hexa2QuadTopologicalMapping', input="@Hexa_Topology", output="@Quad_Topology")
        self.MO = self.root.BeamFEM.addObject('MechanicalObject', src='@Grid', name='MO', template='Vec3d', showObject=True)
        self.root.BeamFEM.addObject('NeoHookeanMaterial', young_modulus=4500, poisson_ratio=0.45, name="StVK")
        self.root.BeamFEM.addObject('HyperelasticForcefield', material="@StVK", template="Hexahedron",
                                    topology='@Hexa_Topology', printLog=True)
        self.root.BeamFEM.addObject('BoxROI', box=p_grid['fixed_box'], name='Fixed_Box')
        self.root.BeamFEM.addObject('FixedConstraint', indices='@Fixed_Box.indices')
        self.CFF = self.root.BeamFEM.addObject('ConstantForceField', name='CFF', showArrowSize='0.1',
                                               forces=[0 for _ in range(3 * self.nb_dof)],
                                               indices=list(iter(range(self.nb_dof))))
        # Visual
        self.root.BeamFEM.addChild('Visual')
        self.root.BeamFEM.Visual.addObject('OglModel', name="oglModel", src='@../Grid', color='white')
        self.root.BeamFEM.Visual.addObject('BarycentricMapping', name="BaryMap2", input='@../MO', output='@./')

    def onSimulationInitDoneEvent(self, event):
        self.inputSize = self.MO.position.value.shape
        self.outputSize = self.MO.position.value.shape

    def onAnimateBeginEvent(self, event):
        self.MO.position.value = self.MO.rest_position.value
        F = np.random.random(3) - np.random.random(3)
        self.CFF.force.value = (3 / np.linalg.norm(F)) * F

    def computeInput(self):
        self.input = copy.copy(self.CFF.forces.value)

    def computeOutput(self):
        self.output = copy.copy(self.MO.position.value - self.MO.rest_position.value)
