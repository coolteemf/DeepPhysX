"""
Prediction scene: NN simulated beam with mouse deformations (launch with the script beamPrediction in FC repository)
"""

import copy

from Example.Beam.MouseForceManager import MouseForceManager
from Example.Beam.NNBeam import NNBeam


class NNBeamMouse(NNBeam):

    def __init__(self, root_node, config, idx_instance=1):
        super(NNBeamMouse, self).__init__(root_node, config, idx_instance)
        self.config = config

    def createBehavior(self, config):
        # Draw sphere on the surface to see on which nodes the forces are applied
        self.sphere = self.root.beamNN.addObject('SphereROI', centers=[0, 0, 0], radii=2, drawSphere=True)

    def onSimulationInitDoneEvent(self, event):
        NNBeam.onSimulationInitDoneEvent(self, event)
        # Mouse manager : avoid too local forces, divide on neighbors
        self.mouse_manager = MouseForceManager(topology=self.grid, max_force=[5.] * 3, surface=self.surface)

    def onAnimateBeginEvent(self, event):
        # Reset position
        self.MO.position.value = self.MO.rest_position.value

    def computeInput(self):
        # Compute the input force to give to the network
        F = copy.copy(self.MO.force.value)
        # Find the node picked with the mouse
        node = self.mouse_manager.find_picked_node(F)
        if node is not None:
            # Scale the force value to fit the range of inputs learned by the network
            F[node] = self.mouse_manager.scale_max_force(force=F[node])
            # Distribute the forces on neighbors (gamma : Gaussian compression / radius : neighborhood)
            F, local = self.mouse_manager.distribute_force(node=node, forces=F, gamma=0.25, radius=30)
            # Show nodes on which forces are applied
            centers = []
            for l in local:
                centers.append(self.MO.position.value[l])
            self.sphere.centers.value = centers
            self.sphere.radii.value = [0.25 for _ in range(len(local) + 1)]
        self.input = F

    def applyPrediction(self, prediction):
        # Add the displacement to the initial position
        u0 = prediction[0]
        self.MO.position.value = u0 + self.MO.rest_position.array()