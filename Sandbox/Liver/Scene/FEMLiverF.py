"""
FEMLiver.py
FEM simulated liver with random forces applied on the visible surface.
Can be launched as a Sofa scene using the 'liverSofa.py' script in Liver repository.
Run with:
    'python3 liverSofa.py 1' or 'python3 liverSofa.py FEMLiver'
"""

from Sandbox.Liver.Scene.BothLiverF import BothLiverF


# Inherit from BothLiver with only setting FEM node
class FEMLiverF(BothLiverF):

    def __init__(self, root_node, config, idx_instance=1):
        super(FEMLiverF, self).__init__(root_node, config, idx_instance)

    def addModels(self, p_liver, p_grid):
        self.createFEM(p_liver, p_grid)
        self.is_created['fem'] = True
