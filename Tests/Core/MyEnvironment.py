import math
import random
import numpy as np

from DeepPhysX.Environment.Environment import Environment


class MyEnvironment(Environment):

    def __init__(self, simulations_per_steps, max_wrong_samples_per_step, idx_instance=1):
        Environment.__init__(self, simulations_per_step=simulations_per_steps,
                             max_wrong_samples_per_step=max_wrong_samples_per_step,
                             idx_instance=idx_instance)
        self.stepCount = 0
        self.inputs = 0
        self.outputs = 0
        self.inputSize, self.outputSize = 1, 1

    def create(self):
        pass

    def reset(self):
        self.stepCount = 0

    def step(self):
        self.stepCount += 1
        # print("{} - Step n°{}".format(self.name, self.stepCount))
        self.inputs = round(random.uniform(-math.pi, math.pi), 2)
        self.outputs = math.sin(self.inputs)

    def getInput(self):
        return self.inputs

    def getOutput(self):
        return self.outputs

    def checkSample(self):
        pass
