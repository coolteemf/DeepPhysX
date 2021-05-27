import copy
from Example.Beam.MouseForceManager import MouseForceManager

from Example.Beam.NNBeam import NNBeam


class NNBeamInteraction(NNBeam):

    def __init__(self, root_node, config, idx_instance=1, training=True):
        super(NNBeamInteraction, self).__init__(root_node, config, idx_instance)
        self.config = config

    def create(self, config):
        NNBeam.create(self, config)
        self.sphere = self.root.BeamNN.addObject('SphereROI', centers=[0, 0, 0], radii=2, drawSphere=True)

    def onSimulationInitDoneEvent(self, event):
        NNBeam.onSimulationInitDoneEvent(self, event)
        self.mouseManager = MouseForceManager(self.grid)

    def onAnimateBeginEvent(self, event):
        self.MO.position.value = self.MO.rest_position.value

    def computeInput(self):
        F = copy.copy(self.MO.force.value)
        node = self.mouseManager.find_picked_node(F)
        if node is not None:
            local = self.mouseManager.find_neighbors(node)
            centers = [self.MO.position.value[node]]
            for l in local:
                centers.append(self.MO.position.value[l])
            self.sphere.centers.value = centers
            self.sphere.radii.value = [0.5 for _ in range(len(local)+1)]

        self.input = F / 800
