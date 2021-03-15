import numpy as np


class Environment:

    def __init__(self, simulations_per_step=1):
        self.simulationPerStep = simulations_per_step
        self.inputs = np.array([])
        self.outputs = np.array([])
        self.inputSize = None
        self.outputSize = None

    def create(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def getInput(self):
        return self.inputs

    def getOutput(self):
        return self.outputs
