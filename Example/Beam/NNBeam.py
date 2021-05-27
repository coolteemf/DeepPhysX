import copy
import numpy as np

from DeepPhysX_Sofa.Environment.SofaBaseEnvironment import SofaBaseEnvironment


class NNBeam(SofaBaseEnvironment):

    def __init__(self, root_node, config, idx_instance=1, training=True):
        super(NNBeam, self).__init__(root_node, config, idx_instance)
        self.config = config

    def create(self, config):
        # Get parameters
        p_grid = config.p_grid
        g_res = p_grid['grid_resolution']
        self.nb_dof = g_res[0] * g_res[1] * g_res[2]

        # Beam FEM
        self.root.addChild('BeamNN')
        self.grid = self.root.BeamNN.addObject('RegularGridTopology', name='Grid', min=p_grid['min'], max=p_grid['max'],
                                               nx=g_res[0], ny=g_res[1], nz=g_res[2])
        self.root.BeamNN.addObject('LegacyStaticODESolver', name='StaticSolver', newton_iterations=0,
                                   correction_tolerance_threshold=1e-6, residual_tolerance_threshold=1e-6,
                                   printLog=False)
        self.root.BeamNN.addObject('HexahedronSetTopologyContainer', name='Hexa_Topology', src='@Grid')
        self.root.BeamNN.addObject('HexahedronSetGeometryAlgorithms', template='Vec3d')
        self.root.BeamNN.addObject('HexahedronSetTopologyModifier')
        self.root.BeamNN.addObject('QuadSetTopologyContainer', name='Quad_Topology', src='@Hexa_Topology')
        self.root.BeamNN.addObject('QuadSetGeometryAlgorithms', template='Vec3d')
        self.root.BeamNN.addObject('QuadSetTopologyModifier')
        self.root.BeamNN.addObject('Hexa2QuadTopologicalMapping', input="@Hexa_Topology", output="@Quad_Topology")
        self.MO = self.root.BeamNN.addObject('MechanicalObject', src='@Grid', name='MO', template='Vec3d',
                                             showObject=True)
        self.root.BeamNN.addObject('BoxROI', box=p_grid['free_box'], name='Free_Box')
        self.CFF = self.root.BeamNN.addObject('ConstantForceField', name='CFF', showArrowSize='0.1',
                                              forces=[0 for _ in range(3 * self.nb_dof - 200)],
                                              indices='@Free_Box.indices')
        # Visual
        self.root.BeamNN.addChild('Visual')
        self.root.BeamNN.Visual.addObject('OglModel', name="oglModel", src='@../Grid', color='white')
        self.root.BeamNN.Visual.addObject('BarycentricMapping', name="BaryMap2", input='@../MO', output='@./')

    def onSimulationInitDoneEvent(self, event):
        self.inputSize = self.MO.position.value.shape
        self.outputSize = self.MO.position.value.shape

    def onAnimateBeginEvent(self, event):
        self.MO.position.value = self.MO.rest_position.value
        F = np.random.random(3) - np.random.random(3)
        self.CFF.force.value = F

    def computeInput(self):
        F = copy.copy(self.MO.force.value)
        self.input = F / 10

    def computeOutput(self):
        self.output = copy.copy(self.MO.position.value - self.MO.rest_position.value)

    def applyPrediction(self, prediction):
        u0 = prediction[0]
        self.MO.position.value = u0 + self.MO.rest_position.array()
