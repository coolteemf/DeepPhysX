import numpy as np

from DeepPhysX_Core.Environment.BaseEnvironment import BaseEnvironment


class Environment(BaseEnvironment):

    def __init__(self, config, idx=1):
        super(Environment, self).__init__(config=config, instance_id=idx)
        self.idx = idx
        self.tensor = None

    def create(self):
        pass

    def step(self):
        print("STep in", self.idx)
        self.tensor = np.random.random((40000, 3))

    def getTensor(self):
        return self.tensor

    def __str__(self):
        return f"Environment n°{self.idx} with tensor {self.tensor}"
