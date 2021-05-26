import copy
import numpy as np

from Example.Beam.NNBeam import NNBeam


class NNBeamInteraction(NNBeam):

    def __init__(self, root_node, config, idx_instance=1, training=True):
        super(NNBeamInteraction, self).__init__(root_node, config, idx_instance)
        self.config = config

    def onAnimateBeginEvent(self, event):
        self.MO.position.value = self.MO.rest_position.value
        F = np.random.random(3) - np.random.random(3)
        self.CFF.force.value = F

    def computeInput(self):
        F = copy.copy(self.MO.force.value)
        F /= 100
        self.input = F

    def computeOutput(self):
        self.output = copy.copy(self.MO.position.value - self.MO.rest_position.value)

    def applyPrediction(self, prediction):
        u0 = prediction[0]
        u0 /= 0.1
        self.MO.position.value = u0 + self.MO.rest_position.array()
