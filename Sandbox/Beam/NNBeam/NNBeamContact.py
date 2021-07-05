"""
NNBeamCollision.py
Neural network simulated beam with a contact with a floor.
Launched with the python script '../beamPrediction.py'.
"""

import os
import copy
import numpy as np

from Sandbox.Beam.NNBeam.NNBeam import NNBeam


class NNBeamContact(NNBeam):

    def __init__(self, root_node, config, idx_instance, visualizer_class=None):
        super(NNBeamContact, self).__init__(root_node, config, idx_instance, visualizer_class)

    def createBehavior(self, config):
        # Beam behaviour : see NNBeam.py
        NNBeam.createBehavior(self, config)
        # Rigid floor
        self.root.addChild("floor")
        filename = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/BeamConfig/floor.obj'
        self.root.floor.addObject('MeshObjLoader', name="loader", filename=filename, translation=[0, -3, 0])
        self.root.floor.addObject('MeshTopology', src='@loader')
        self.root.floor.addObject('MechanicalObject', src='@loader', template='Vec3d', showObject=False)

    def createVisual(self, config):
        # Visual style of the scene
        self.root.addObject('VisualStyle', displayFlags="showCollisionModels hideVisualModels hideBehavior")

    def createCollision(self, config):
        # Collision pipeline
        self.root.addObject('DefaultPipeline', depth=8)
        self.root.addObject('BruteForceDetection')
        self.root.addObject('MinProximityIntersection', alarmDistance=2, contactDistance=0.75)
        self.root.addObject('DefaultContactManager', name="Response", response="default")
        # Beam collision model
        self.root.beamNN.addChild('collision')
        p_grid = config.p_grid
        g_res = p_grid['grid_resolution']
        self.root.beamNN.collision.addObject('RegularGridTopology', name='Grid', min=p_grid['grid_min'],
                                             max=p_grid['grid_max'], nx=g_res[0], ny=g_res[1], nz=g_res[2])
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
        # Apply constant force field on the top
        self.root.beamNN.removeObject(self.box)
        self.box = self.root.beamNN.addObject('BoxROI', name='ForceBox', drawBoxes=True, drawSize=1,
                                              box=self.config.p_grid['top_box'])
        self.box.init()
        indices = list(self.box.indices.value)
        indices = list(set(indices).intersection(set(self.idx_surface)))
        F = np.array([0.0, -0.5, 0.0])
        self.root.beamNN.removeObject(self.CFF)
        self.CFF = self.root.beamNN.addObject('ConstantForceField', name='CFF', showArrowSize='1',
                                              indices=indices, force=list(F))
        self.CFF.init()

    def onAnimateBeginEvent(self, event):
        pass

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
        # Dynamical factor
        U = ((1e-3 / self.nb_node) / self.root.dt.value ** 2) * prediction[0]
        # Add the displacement to the initial position
        self.MO.position.value = self.MO.position.array() + U
        # The mechanical and collision models are the same
        self.CMO.position.value = self.MO.position.value
