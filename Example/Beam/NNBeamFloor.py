import os
import copy
import numpy as np

from Example.Beam.NNBeam import NNBeam
from Example.Beam.MouseForceManager import MouseForceManager


class NNBeamFloor(NNBeam):

    def __init__(self, root_node, config, idx_instance):
        super(NNBeamFloor, self).__init__(root_node, config, idx_instance)

    def createBehavior(self, config):
        # Beam behaviour
        NNBeam.createBehavior(self, config)
        self.CFF.showArrowSize.value = 0.0
        # Floor behaviour
        self.root.addChild("floor")
        filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'floor.obj')
        self.root.floor.addObject('MeshObjLoader', name="loader", filename=filename)
        self.root.floor.addObject('MeshTopology', src='@loader')
        self.root.floor.addObject('MechanicalObject', src='@loader', template='Vec3d', showObject=False)

    def createVisual(self, config):
        self.root.addObject('VisualStyle', displayFlags="showCollisionModels hideVisualModels hideBehavior")

    def createCollision(self, config):
        # Collision pipeline
        self.root.addObject('DefaultPipeline', depth=8)
        self.root.addObject('BruteForceDetection')
        self.root.addObject('MinProximityIntersection', alarmDistance=2, contactDistance=0.5)
        self.root.addObject('DefaultContactManager', name="Response", response="default")
        # Beam collision model
        self.root.beamNN.addChild('collision')
        p_grid = config.p_grid
        g_res = p_grid['grid_resolution']
        self.root.beamNN.collision.addObject('RegularGridTopology', name='Grid', min=p_grid['min'], max=p_grid['max'],
                                             nx=g_res[0], ny=g_res[1], nz=g_res[2])
        self.CMO = self.root.beamNN.collision.addObject('MechanicalObject', src='@Grid', name='MO', template='Vec3d',
                                                        showObject=False)
        self.root.beamNN.collision.addObject('TriangleCollisionModel', bothSide='True')
        self.root.beamNN.collision.addObject('LineCollisionModel')
        self.root.beamNN.collision.addObject('PointCollisionModel')
        # Floor collision model
        self.root.floor.addObject('TriangleCollisionModel', bothSide=True, moving=False, simulated=False,
                                  contactStiffness=1)
        self.root.floor.addObject('LineCollisionModel', moving=False, simulated=False)
        self.root.floor.addObject('PointCollisionModel', moving=False, simulated=False)

    def onSimulationInitDoneEvent(self, event):
        NNBeam.onSimulationInitDoneEvent(self, event)
        self.mouseManager = MouseForceManager(self.grid, [2.] * 3, self.surface)
        self.max_force = np.zeros_like(self.MO.force.value)

    def onAnimateBeginEvent(self, event):
        # self.MO.position.value = self.MO.rest_position.value
        F = np.array([0.0, -0.2, 0.0])
        self.CFF.force.value = F

    def computeInput(self):
        # Get applied forces
        f = copy.copy(self.MO.force.value)
        F = np.zeros_like(f)
        F[self.idx_surface] = f[self.idx_surface]
        # Get collision forces
        c = copy.copy(self.CMO.force.value)
        C = np.zeros_like(c)
        C[self.idx_surface] = c[self.idx_surface]
        # Add forces
        self.input = F + C

    def applyPrediction(self, prediction):
        u0 = ((1e-3 / self.nb_dof) / self.root.dt.value ** 2) * prediction[0]
        self.MO.position.value = self.MO.position.array() + u0
        self.CMO.position.value = self.MO.position.value
