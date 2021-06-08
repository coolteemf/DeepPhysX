import copy
import numpy as np
import os

from DeepPhysX_Sofa.Environment.SofaBaseEnvironment import SofaBaseEnvironment


class NNBeam(SofaBaseEnvironment):

    def __init__(self, root_node, config, idx_instance=1, training=True):
        super(NNBeam, self).__init__(root_node, config, idx_instance)
        self.config = config
        self.nbSteps = 0

    def create(self, config):
        # Get parameters
        p_grid = config.p_grid
        g_res = p_grid['grid_resolution']
        self.nb_dof = g_res[0] * g_res[1] * g_res[2]

        # Beam FEM
        self.root.addChild('BeamNN')
        self.root.BeamNN.addObject('LegacyStaticODESolver', name='StaticSolver', newton_iterations=0, printLog=False)
        # Geometry
        self.grid = self.root.BeamNN.addObject('RegularGridTopology', name='Grid', min=p_grid['min'], max=p_grid['max'],
                                               nx=g_res[0], ny=g_res[1], nz=g_res[2])
        self.root.BeamNN.addObject('HexahedronSetTopologyContainer', name='HexaTopology', src='@Grid')
        self.root.BeamNN.addObject('HexahedronSetGeometryAlgorithms', template='Vec3d')
        self.root.BeamNN.addObject('HexahedronSetTopologyModifier')
        self.surface = self.root.BeamNN.addObject('QuadSetTopologyContainer', name='QuadTopology', src='@HexaTopology')
        self.root.BeamNN.addObject('QuadSetGeometryAlgorithms', template='Vec3d')
        self.root.BeamNN.addObject('QuadSetTopologyModifier')
        self.root.BeamNN.addObject('Hexa2QuadTopologicalMapping', input="@HexaTopology", output="@QuadTopology")
        # Behavior
        self.createBehavior(p_grid)
        # Visual
        self.createVisual()
        # Collision
        self.createCollision()

    def createBehavior(self, p_grid):
        self.MO = self.root.BeamNN.addObject('MechanicalObject', src='@Grid', name='MO', template='Vec3d',
                                             showObject=True)
        self.root.BeamNN.addObject('BoxROI', box=p_grid['free_box'], name='FreeBox')
        self.CFF = self.root.BeamNN.addObject('ConstantForceField', name='CFF', showArrowSize='2.5',
                                              force=[0, 0, 0], indices='@FreeBox.indices')

    def createVisual(self):
        self.root.BeamNN.addChild('Visual')
        file = os.pardir + '/Beam/Beam_really_smooth.obj'
        self.root.BeamNN.Visual.addObject('MeshObjLoader', name="Mesh", filename=file,
                                          translation=[100, 12.5, 12.5])
        self.root.BeamNN.Visual.addObject('OglModel', name="oglModel", src='@Mesh', color='white')
        # self.root.BeamNN.Visual.addObject('OglModel', name="oglModel", src='@../Grid', color='white')
        self.root.BeamNN.Visual.addObject('BarycentricMapping', name="BaryMap2", input='@../MO', output='@./')

    def createCollision(self):
        # self.root.BeamNN.addChild('Collision')
        # self.root.BeamNN.Collision.addObject('TriangleSetTopologyContainer', name='TriTopology', src='@../QuadTopology')
        # self.root.BeamNN.Collision.addObject('TriangleSetGeometryAlgorithms', template='Vec3d')
        # self.root.BeamNN.Collision.addObject('TriangleSetTopologyModifier')
        # self.root.BeamNN.Collision.addObject('Quad2TriangleTopologicalMapping', input="@../QuadTopology",
        #                                      output="@TriTopology")
        # self.root.BeamNN.Collision.addObject('OglModel', name="oglModel")
        # self.root.BeamNN.Collision.addObject('BarycentricMapping', name='debug', input='@../MO', output='@./')
        # self.root.BeamNN.Collision.addObject('TriangleCollisionModel')
        # self.root.BeamNN.Collision.addObject('LineCollisionModel')
        # self.root.BeamNN.Collision.addObject('PointCollisionModel')
        pass

    def onSimulationInitDoneEvent(self, event):
        self.inputSize = self.MO.position.value.shape
        self.outputSize = self.MO.position.value.shape
        self.idx = self.surface.quads.value.reshape(-1)

    def onAnimateBeginEvent(self, event):
        self.MO.position.value = self.MO.rest_position.value
        t, t2 = np.cos(2.0*np.pi*self.nbSteps/100), np.sin(2.0*np.pi*self.nbSteps/100)
        F = np.array([0.0, np.sin(t)*np.sin(t2), np.sin(t)*np.cos(t2)])
        # F = np.random.rand(3) - np.random.rand(3)
        self.CFF.force.value = F

    def onAnimateEndEvent(self, event):
        self.nbSteps += 1

    def computeInput(self):
        f = copy.copy(self.MO.force.value)
        F = np.zeros_like(f)
        F[self.idx] = f[self.idx]
        self.input = F / 5

    def computeOutput(self):
        self.output = copy.copy(self.MO.position.value - self.MO.rest_position.value)

    def applyPrediction(self, prediction):
        u0 = prediction[0]
        self.MO.position.value = u0 + self.MO.rest_position.array()
