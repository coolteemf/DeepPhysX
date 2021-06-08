import copy
import numpy as np

from DeepPhysX_Sofa.Environment.SofaBaseEnvironment import SofaBaseEnvironment


class NNBeamBall(SofaBaseEnvironment):

    def __init__(self, root_node, config, idx_instance=1, training=True):
        super(NNBeamBall, self).__init__(root_node, config, idx_instance)
        self.config = config

    def create(self, config):
        # Get parameters
        p_grid = config.p_grid
        g_res = p_grid['grid_resolution']
        self.nb_dof = g_res[0] * g_res[1] * g_res[2]

        # Beam NN
        self.root.addChild('BeamNN')
        self.grid = self.root.BeamNN.addObject('RegularGridTopology', name='Grid', min=p_grid['min'], max=p_grid['max'],
                                               nx=g_res[0], ny=g_res[1], nz=g_res[2])
        self.root.BeamNN.addObject('LegacyStaticODESolver', name='StaticSolver', newton_iterations=0, printLog=False)
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
                                              force=[0, 0, 0], indices='@Free_Box.indices')
        # Visual
        self.root.BeamNN.addChild('Visual')
        self.root.BeamNN.Visual.addObject('OglModel', name="oglModel", src='@../Grid', color='white')
        self.root.BeamNN.Visual.addObject('BarycentricMapping', name="BaryMap2", input='@../MO', output='@./')
        # Collision
        self.root.BeamNN.addChild('Collision')
        self.root.BeamNN.Collision.addObject('RegularGridTopology', name='Grid', min=p_grid['min'], max=p_grid['max'],
                                             nx=g_res[0], ny=g_res[1], nz=g_res[2])
        self.root.BeamNN.Collision.addObject('MechanicalObject', src='@Grid', template='Vec3d', showObject=False)
        self.root.BeamNN.Collision.addObject('PointCollisionModel')
        # self.root.BeamNN.Collision.addObject('LineCollisionModel')
        self.root.BeamNN.Collision.addObject('IdentityMapping', input='@../MO', output='@./')

        # Ball FEM
        self.root.addChild('BallFEM')
        self.root.BallFEM.addObject('EulerImplicitSolver', name='DynamicSolver')
        self.root.BallFEM.addObject('ConjugateGradientSolver', name='LinearSolver')
        self.root.BallFEM.addObject('MechanicalObject', name='MO', showObject=True, template='Rigid3d',
                                    position=[50, 50, 50, 0, 0, 0, 1])
        self.root.BallFEM.addObject('UniformMass', totalMass=1000)
        # Collision
        self.root.BallFEM.addChild('Collision')
        self.root.BallFEM.Collision.addObject('MeshObjLoader', name="loader", filename="mesh/ball.obj",
                                              scale=5)
        self.root.BallFEM.Collision.addObject('MechanicalObject', src='@loader', template='Vec3d', showObject=False)
        self.root.BallFEM.Collision.addObject('PointCollisionModel')
        self.root.BallFEM.Collision.addObject('RigidMapping')

    def onSimulationInitDoneEvent(self, event):
        self.inputSize = self.MO.position.value.shape
        self.outputSize = self.MO.position.value.shape

    def onAnimateBeginEvent(self, event):
        self.MO.position.value = self.MO.rest_position.value
        F = np.random.random(3) - np.random.random(3)
        self.CFF.force.value = F

    def computeInput(self):
        F = copy.copy(self.MO.force.value)
        self.input = F / 2

    def computeOutput(self):
        self.output = copy.copy(self.MO.position.value - self.MO.rest_position.value)

    def applyPrediction(self, prediction):
        u0 = prediction[0]
        self.MO.position.value = u0 + self.MO.rest_position.array()
