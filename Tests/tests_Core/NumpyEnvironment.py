import numpy as np

from DeepPhysX_Core.Environment.BaseEnvironment import BaseEnvironment


class NumpyEnvironment(BaseEnvironment):

    def __init__(self, config, idx_instance=1, visualizer_class=None):
        BaseEnvironment.__init__(self, config=config, idx_instance=idx_instance)
        self.idx_step = 0
        self.a = np.random.randn()
        self.b = np.random.randn()
        self.c = np.random.randn()
        self.d = np.random.randn()

    def create(self, config):
        self.input = np.random.randn(1)
        self.input_size = self.input.size
        self.output = self.input
        self.output_size = self.output.size

    def step(self):
        self.idx_step += 1

    def computeInput(self):
        self.input = np.random.randn(1).round(2)

    def computeOutput(self):
        x = self.input
        self.output = self.a + self.b * x + self.c * x ** 2 - self.d * x ** 3

    def reset(self):
        self.idx_step = 0
