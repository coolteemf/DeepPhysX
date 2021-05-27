import copy
import numpy as np

from Example.Beam.BothBeams import BothBeams


class BothBeamsInteraction(BothBeams):

    def __init__(self, root_node, config, idx_instance=1, training=True):
        super(BothBeamsInteraction, self).__init__(root_node, config, idx_instance)
        self.config = config

    def onAnimateBeginEvent(self, event):
        self.femMO.position.value = self.femMO.rest_position.value
        self.MO.position.value = self.MO.rest_position.value

    def computeInput(self):
        F = copy.copy(self.femMO.force.value)
        print("Norm force:", np.linalg.norm(F))
        F /= 0.001
        self.input = F

    def applyPrediction(self, prediction):
        u0 = prediction[0]
        print("Norm displacement:", np.linalg.norm(u0))
        u0 /= 0.1
        self.MO.position.value = u0 + self.MO.rest_position.array()
