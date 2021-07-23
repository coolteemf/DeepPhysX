"""
NNLiver.py
Neural Network simulated liver with random forces applied on the visible surface.
Must be launched through DeepPhysX running pipeline using the 'liverUnet.py' script in Liver repository.
Run with:
    'python3 liverUnet.py 1' or 'python3 liverUnet.py NNLiver'
"""

import copy
import numpy as np

from Sandbox.Liver.Scene.BothLiver import BothLiver


# Inherit from SofaEnvironment which allow to implement and create a Sofa scene in the DeepPhysX_Core pipeline
class NNLiver(BothLiver):

    def __init__(self, root_node, config, idx_instance=1,):
        super(NNLiver, self).__init__(root_node, config, idx_instance)

    def addModels(self, p_liver, p_grid):
        self.createNN(p_liver, p_grid)

    def computeOutput(self):
        self.output = np.zeros((self.nb_nodes_regular_grid, 3), dtype=np.double)
