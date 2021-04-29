import numpy as np


class BaseEnvironment:

    def __init__(self, config, idx_instance=1):

        self.name = "Environment n°{}".format(idx_instance)
        self.simulationsPerStep = config.simulations_per_step
        self.maxWrongSamplesPerStep = config.max_wrong_samples_per_step

        self.inputs = np.array([])
        self.outputs = np.array([])
        self.inputSize = None
        self.outputSize = None

        self.description = ""
        self.descriptionName = "CORE Environment"

        self.create(config)

    def create(self, config):
        print("WARNING: You have to implement environment create() method.")

    def reset(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def getInput(self):
        return self.inputs

    def getOutput(self):
        return self.outputs

    def checkSample(self):
        raise NotImplementedError

    def getDescription(self):
        if len(self.description) == 0:
            self.description += "\n{}\n".format(self.descriptionName)
            self.description += "   Name: {}\n".format(self.name)
            self.description += "   Simulations per step: {}\n".format(self.simulationsPerStep)
            self.description += "   Max wrong samples per step: {}\n".format(self.maxWrongSamplesPerStep)
            self.description += "   Inputs, size: {}\n".format(self.inputs, self.inputSize)
            self.description += "   Outputs, size: {}\n".format(self.outputs, self.outputSize)
        return self.description
