import Sofa
import Sofa.Simulation
import numpy as np
import time


class Environment(Sofa.Core.Controller):

    def __init__(self, root, idx=1, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.root = root
        self.idx = idx
        self.tensor = None
        # print(self)

    def step(self):
        Sofa.Simulation.animate(self.root, self.root.dt.value)
        # time.sleep(10*np.random.randn())

    def onAnimateBeginEvent(self, event):
        self.tensor = np.random.randn(3)

    def getTensor(self):
        return self.tensor

    def __str__(self):
        return f"Environment n°{self.idx} with tensor {self.tensor}"
