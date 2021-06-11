import numpy as np

from DeepPhysX.Environment.BaseEnvironment import BaseEnvironment


class NumpyEnvironment(BaseEnvironment):

    def __init__(self, config, idx_instance=1):
        BaseEnvironment.__init__(self, config=config, idx_instance=idx_instance)
        self.idx_step = 0

    def create(self, config):
        self.input = np.random.randn(1)
        self.input_size = self.input.shape
        self.output = self.input
        self.output_size = self.output.shape

    def step(self):
        self.idx_step += 1
        # print("{} - Step n°{}".format(self.name, self.stepCount))

    def computeInput(self):
        self.input = np.random.randn(1).round(2)

    def computeOutput(self):
        x = self.input
        self.output = 0.3 + 0.1 * x + 0.5 * x ** 2 - 0.15 * x ** 3

    def reset(self):
        self.idx_step = 0
