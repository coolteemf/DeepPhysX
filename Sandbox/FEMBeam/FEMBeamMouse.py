"""
FEMBeamMouse.py
FEM simulated beam with deformations applied with the mouse in the GUI.
Can be launched as a Sofa scene using the 'runSofa.py' script in this repository.
"""

from Sandbox.FEMBeam.FEMBeam import FEMBeam


class FEMBeamMouse(FEMBeam):

    def __init__(self, root_node, config, idx_instance=1):
        FEMBeam.__init__(self, root_node, config, idx_instance)

    def onAnimateBeginEvent(self, event):
        # Only reset the position of the mechanical object when the mouse drop the beam
        self.MO.position.value = self.MO.rest_position.value
        # Todo: add mouse manager
