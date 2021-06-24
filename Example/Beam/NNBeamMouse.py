import copy
import numpy as np
from Example.Beam.MouseForceManager import MouseForceManager

from Example.Beam.NNBeam import NNBeam


class NNBeamMouse(NNBeam):

    def __init__(self, root_node, config, idx_instance=1, training=True):
        super(NNBeamMouse, self).__init__(root_node, config, idx_instance)
        self.config = config

    def createBehavior(self, config):
        self.sphere = self.root.beamNN.addObject('SphereROI', centers=[0, 0, 0], radii=2, drawSphere=True)

    def onSimulationInitDoneEvent(self, event):
        NNBeam.onSimulationInitDoneEvent(self, event)
        self.mouseManager = MouseForceManager(self.grid, [2.] * 3, self.surface)

    def onAnimateBeginEvent(self, event):
        self.MO.position.value = self.MO.rest_position.value

    def computeInput(self):
        F = copy.copy(self.MO.force.value)
        node = self.mouseManager.find_picked_node(F)
        if node is not None:
            F[node] = self.mouseManager.scale_max_force(F[node])
            F, local = self.mouseManager.distribute_force(node, F, 0.05, 15)
            centers = []
            for l in local:
                centers.append(self.MO.position.value[l])
            self.sphere.centers.value = centers
            self.sphere.radii.value = [0.25 for _ in range(len(local) + 1)]
        norm = np.linalg.norm(F) if np.linalg.norm(F) != 0 else 1
        self.input = F / norm

    def applyPrediction(self, prediction):
        u0 = prediction[0] * 10
        self.MO.position.value = u0 + self.MO.rest_position.array()
