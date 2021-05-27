from Example.Beam.FEMBeam import FEMBeam


class FEMBeamInteraction(FEMBeam):

    def __init__(self, root_node, config, idx_instance=1, training=True):
        super(FEMBeamInteraction, self).__init__(root_node, config, idx_instance)

    def onAnimateBeginEvent(self, event):
        self.MO.position.value = self.MO.rest_position.value
