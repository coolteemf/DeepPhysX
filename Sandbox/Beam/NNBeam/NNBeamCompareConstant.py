"""
NNBeamCompare.py
Neural network simulated beam with constant deformations compared to the FEM ground truth.
Launched with the python script '../beamPrediction.py'.
"""

import numpy as np

from Sandbox.Beam.NNBeam.NNBeamCompare import NNBeamCompare


class NNBeamCompareConstant(NNBeamCompare):

    def __init__(self, root_node, config, idx_instance=1, visualizer_class=None):
        super(NNBeamCompareConstant, self).__init__(root_node, config, idx_instance, visualizer_class)

    def onSimulationInitDoneEvent(self, event):
        NNBeamCompare.onSimulationInitDoneEvent(self, event)

        # Remove the unused objects from the parent class
        self.root.beamFEM.removeObject(self.femBox)
        self.root.beamFEM.removeObject(self.femCFF)
        self.root.beamNN.removeObject(self.nnBox)
        self.root.beamNN.removeObject(self.nnCFF)

        # Caution : the intersection between box and surface needs to be non empty
        x_min, x_max = 80, 90               # x_grid in [0., 100.] + margin of the box
        y_min, y_max = 10, 15.5             # y_grid in [0., 15.] + margin of the box
        z_min, z_max = -0.5, 15.5           # z_grid in [0., 15.] + margin of the box
        F = np.array([1., -0.5, -0.75])     # F_i in [-1., 1.]

        # Set a new bounding box
        # self.root.beamFEM.removeObject(self.femBox)
        self.femBox = self.root.beamFEM.addObject('BoxROI', name='ForceBox', drawBoxes=True, drawSize=1,
                                                  box=[x_min, y_min, z_min, x_max, y_max, z_max])
        self.femBox.init()
        # Get the intersection with the surface
        indices = list(self.femBox.indices.value)
        indices = list(set(indices).intersection(set(self.idx_surface)))
        # Same box for NN simulated beam
        # self.root.beamNN.removeObject(self.nnBox)
        z_min -= 2 * self.config.p_grid['grid_max'][2]
        z_max -= 2 * self.config.p_grid['grid_max'][2]
        self.nnBox = self.root.beamNN.addObject('BoxROI', name='ForceBox', drawBoxes=True, drawSize=1,
                                                box=[x_min, y_min, z_min, x_max, y_max, z_max])
        self.nnBox.init()
        # Create a random constant force field for the nodes in the bbox
        F = (10 / np.linalg.norm(F)) * F
        # Set a new constant force field (variable number of indices)
        # self.root.beamFEM.removeObject(self.femCFF)
        self.femCFF = self.root.beamFEM.addObject('ConstantForceField', name='CFF', showArrowSize='1',
                                                  indices=indices, force=list(F))
        self.femCFF.init()
        # self.root.beamNN.removeObject(self.nnCFF)
        self.nnCFF = self.root.beamNN.addObject('ConstantForceField', name='CFF', showArrowSize='1',
                                                indices=indices, force=list(F))
        self.nnCFF.init()

    def onAnimateBeginEvent(self, event):
        # Reset position
        self.femMO.position.value = self.femMO.rest_position.value
        self.nnMO.position.value = self.nnMO.rest_position.value
