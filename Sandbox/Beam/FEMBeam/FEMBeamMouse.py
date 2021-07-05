"""
FEMBeamMouse.py
FEM simulated beam with deformations applied with the mouse in the GUI.
Can be launched as a Sofa scene using the 'runSofa.py' script in this repository.
"""

from Sandbox.Beam.FEMBeam.FEMBeam import FEMBeam


class FEMBeamMouse(FEMBeam):

    def __init__(self, root_node, config, idx_instance=1, visualizer_class=None):
        FEMBeam.__init__(self, root_node, config, idx_instance, visualizer_class)

    def onAnimateBeginEvent(self, event):
        # Only reset the position of the mechanical object when the mouse drop the beam
        self.MO.position.value = self.MO.rest_position.value
        # Todo: add mouse manager
