import copy
import numpy as np
import os

from DeepPhysX_Sofa.Environment.SofaEnvironment import SofaEnvironment


class NNBeam(SofaEnvironment):

    def __init__(self, root_node, config, idx_instance=1, training=True):
        super(NNBeam, self).__init__(root_node, config, idx_instance)
        self.config = config
        self.nbSteps = 0

    def create(self, config):
        # Grid resolution
        p_grid = config.p_grid
        g_res = p_grid['grid_resolution']
        self.nb_dof = g_res[0] * g_res[1] * g_res[2]

        # Beam
        self.root.addChild('beamNN')
        self.root.beamNN.addObject('LegacyStaticODESolver', name='StaticSolver', newton_iterations=0, printLog=False)
        self.grid = self.root.beamNN.addObject('RegularGridTopology', name='Grid', min=p_grid['min'], max=p_grid['max'],
                                               nx=g_res[0], ny=g_res[1], nz=g_res[2])
        self.MO = self.root.beamNN.addObject('MechanicalObject', src='@Grid', name='MO', template='Vec3d',
                                             showObject=False)
        self.root.beamNN.addObject('HexahedronSetTopologyContainer', name='HexaTopology', src='@Grid')
        self.root.beamNN.addObject('HexahedronSetGeometryAlgorithms', template='Vec3d')
        self.root.beamNN.addObject('HexahedronSetTopologyModifier')
        self.surface = self.root.beamNN.addObject('QuadSetTopologyContainer', name='QuadTopology', src='@HexaTopology')
        self.root.beamNN.addObject('QuadSetGeometryAlgorithms', template='Vec3d')
        self.root.beamNN.addObject('QuadSetTopologyModifier')
        self.root.beamNN.addObject('Hexa2QuadTopologicalMapping', input="@HexaTopology", output="@QuadTopology")
        # Behavior
        self.createBehavior(config)
        # Visual
        self.createVisual(config)
        # Collision
        self.createCollision(config)

    def createBehavior(self, config):
        p_grid = config.p_grid
        self.root.beamNN.addObject('BoxROI', box=p_grid['free_box'], name='FreeBox')
        self.CFF = self.root.beamNN.addObject('ConstantForceField', name='CFF', showArrowSize='2.5', force=[0, 0, 0],
                                              indices='@FreeBox.indices')

    def createVisual(self, config):
        self.root.addObject('VisualStyle', displayFlags="showVisualModels")
        self.root.beamNN.addChild('visual')
        # file = os.pardir + '/Beam/Beam_really_smooth.obj'
        # self.root.beamNN.visual.addObject('MeshObjLoader', name="Mesh", filename=file, translation=[100, 12.5, 12.5])
        # self.root.beamNN.visual.addObject('OglModel', name="oglModel", src='@Mesh', color='white')
        self.root.beamNN.visual.addObject('OglModel', name="oglModel", src='@../Grid', color='white')
        self.root.beamNN.visual.addObject('BarycentricMapping', name="BaryMap2", input='@../MO', output='@./')

    def createCollision(self, config):
        pass

    def onSimulationInitDoneEvent(self, event):
        self.input_size = self.MO.position.value.shape
        self.output_size = self.MO.position.value.shape
        self.idx_surface = self.surface.quads.value.reshape(-1)

    # def onAnimateBeginEvent(self, event):
    #     # self.MO.position.value = self.MO.rest_position.value
    #     t, t2 = np.cos(2.0*np.pi*self.nbSteps/200), np.sin(2.0*np.pi*self.nbSteps/200)
    #     # F = np.array([0.0, np.sin(t)*np.sin(t2), np.sin(t)*np.cos(t2)])
    #     F = np.array([0.0, np.sin(t) * np.sin(t2), 0.0])
    #     self.CFF.force.value = F

    def onAnimateEndEvent(self, event):
        self.nbSteps += 1

    def computeInput(self):
        f = copy.copy(self.MO.force.value)
        F = np.zeros_like(f)
        F[self.idx_surface] = f[self.idx_surface]
        self.input = F

    def computeOutput(self):
        self.output = copy.copy(self.MO.position.value - self.MO.rest_position.value)

    def applyPrediction(self, prediction):
        u0 = prediction[0]
        self.MO.position.value = u0 + self.MO.rest_position.array()
