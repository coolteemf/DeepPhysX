"""
NNBeamCollision.py
Neural network simulated beam with a collision with a falling sphere.
Launched with the python script '../beamPrediction.py'.
"""

import copy
import numpy as np

from Sandbox.Beam.BeamConfig.MouseForceManager import MouseForceManager
from Sandbox.Beam.NNBeam.NNBeam import NNBeam


class NNBeamCollision(NNBeam):

    def __init__(self, root_node, config, idx_instance=1, visualizer_class=None):
        super(NNBeamCollision, self).__init__(root_node, config, idx_instance, visualizer_class)
        self.config = config
        self.force_step = 0

    def createBehavior(self, config):
        # Ball behavior
        self.root.addChild('sphere')
        self.root.sphere.addObject('EulerImplicitSolver', name='DynamicSolver')
        self.root.sphere.addObject('CGLinearSolver', name='LinearSolver', iterations=100, tolerance=1.0e-8,
                                   threshold=1.0e-8)
        self.root.sphere.addObject('MechanicalObject', name='MO', showObject=True, template='Rigid3d',
                                   position=[90, 40, 10, 0, 0, 0, 1])
        self.root.sphere.addObject('UniformMass', totalMass=0.1)

    def createVisual(self, config):
        # Visual style of the scene
        self.root.addObject('VisualStyle', displayFlags="showCollisionModels hideVisualModels")

    def createCollision(self, config):
        # Collision pipeline
        self.root.addObject('DefaultPipeline', depth=8)
        self.root.addObject('BruteForceDetection')
        self.root.addObject('MinProximityIntersection', alarmDistance=1, contactDistance=0.1)
        self.root.addObject('DefaultContactManager', name="Response", response="default")
        # Beam collision model
        self.root.beamNN.addChild('collision')
        p_grid = config.p_grid
        g_res = p_grid['grid_resolution']
        self.root.beamNN.collision.addObject('RegularGridTopology', name='Grid', min=p_grid['grid_min'],
                                             max=p_grid['grid_max'], nx=g_res[0], ny=g_res[1], nz=g_res[2])
        self.CMO = self.root.beamNN.collision.addObject('MechanicalObject', src='@Grid', name='MO', template='Vec3d',
                                                        showObject=False)
        self.root.beamNN.collision.addObject('TriangleCollisionModel', bothSide=True)
        self.root.beamNN.collision.addObject('LineCollisionModel')
        self.root.beamNN.collision.addObject('PointCollisionModel')
        # Ball collision model
        self.root.sphere.addChild('collision')
        self.root.sphere.collision.addObject('MeshObjLoader', name="loader", filename="mesh/sphere_05.obj",
                                             scale=0.05)
        self.root.sphere.collision.addObject('MeshTopology', src='@loader')
        self.root.sphere.collision.addObject('MechanicalObject', src='@loader', template='Vec3d', showObject=False)
        self.root.sphere.collision.addObject('PointCollisionModel')
        self.root.sphere.collision.addObject('TriangleCollisionModel')
        self.root.sphere.collision.addObject('LineCollisionModel')
        self.root.sphere.collision.addObject('RigidMapping')

    def onSimulationInitDoneEvent(self, event):
        NNBeam.onSimulationInitDoneEvent(self, event)
        self.mouse_manager = MouseForceManager(topology=self.grid, max_force=[5.0] * 3, surface=self.surface)
        self.last_U = np.zeros((self.nb_node, 3))

    def onAnimateBeginEvent(self, event):
        pass

    def computeInput(self):
        # Compute the input force to give to the network
        F = copy.copy(self.CMO.force.value)
        # Find the node touched by the sphere
        node = self.mouse_manager.find_picked_node(F)
        if node is not None:
            # Scale the force value to fit the range of inputs learned by the network
            F[node] = self.mouse_manager.scale_max_force(forces=F[node])
            # Distribute the forces on neighbors (gamma : Gaussian compression / radius : neighborhood)
            F, local = self.mouse_manager.distribute_force(node=node, forces=F, gamma=0.25, radius=5)
        # Interpolate force
        if np.linalg.norm(F) > 0.0:
            self.force_step = 0.25
            self.sum_forces = copy.copy(F)
            final_force = np.multiply(self.sum_forces, self.force_step)
        else:
            if 0 < self.force_step < 0.999:
                self.force_step += 0.25
                final_force = np.multiply(self.sum_forces, self.force_step)
            else:
                final_force = np.zeros((self.nb_node, 3))
        self.input = final_force

    def applyPrediction(self, prediction):
        U = prediction[0]
        # Interpolate deformation
        if np.linalg.norm(self.input) > 0.0:
            self.last_U = U
            self.MO.position.value = self.MO.rest_position.value + U
            self.CMO.position.value = self.CMO.rest_position.array() + U
            # print("New input force || f || = ", np.linalg.norm(self.input), " -- || last U || =",
            #       np.linalg.norm(self.last_U))
        else:
            self.last_U = np.multiply(self.last_U, 0.75)
            self.CMO.position.value = self.last_U + self.CMO.rest_position.array()
            self.MO.position.value = self.MO.rest_position.value + self.last_U
            # if np.linalg.norm(self.last_U) > 0.0001:
            #     print("Interpolating to rest position -- || last U || =", np.linalg.norm(self.last_U))
