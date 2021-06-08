import copy
from Example.Beam.MouseForceManager import MouseForceManager

from Example.Beam.NNBeam import NNBeam


class NNBeamInteraction(NNBeam):

    def __init__(self, root_node, config, idx_instance=1, training=True):
        super(NNBeamInteraction, self).__init__(root_node, config, idx_instance)
        self.config = config

    def onSimulationInitDoneEvent(self, event):
        NNBeam.onSimulationInitDoneEvent(self, event)
        self.mouseManager = MouseForceManager(self.grid, [2, 2, 2])

    def onAnimateBeginEvent(self, event):
        self.MO.position.value = self.MO.rest_position.value

    def computeInput(self):
        F = copy.copy(self.MO.force.value)
        node = self.mouseManager.find_picked_node(F)
        if node is not None:
            F[node] = self.mouseManager.scale_max_force(F[node])
            F, local = self.mouseManager.distribute_force(node, F, 2.5, 15)
        self.input = F

    def applyPrediction(self, prediction):
        u0 = prediction[0] * 10
        self.MO.position.value = u0 + self.MO.rest_position.array()
