import numpy as np


class BaseEnvironment:

    def __init__(self, config, idx_instance=1):

        self.name = "Environment n°{}".format(idx_instance)
        self.simulationsPerStep = config.simulations_per_step
        self.maxWrongSamplesPerStep = config.max_wrong_samples_per_step

        self.input = np.array([])
        self.output = np.array([])
        self.inputSize = None
        self.outputSize = None

        self.description = ""
        self.descriptionName = "CORE Environment"

        self.create(config)

    def create(self, config):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def computeInput(self):
        raise NotImplementedError

    def computeOutput(self):
        raise NotImplementedError

    def transformInputs(self, inputs):
        return inputs

    def transformOutputs(self, outputs):
        return outputs

    def checkSample(self, check_input=True, check_output=True):
        return True

    def getInput(self):
        return self.input

    def getOutput(self):
        return self.output

    def getDescription(self):
        if len(self.description) == 0:
            self.description += "\n{}\n".format(self.descriptionName)
            self.description += "   Name: {}\n".format(self.name)
            self.description += "   Simulations per step: {}\n".format(self.simulationsPerStep)
            self.description += "   Max wrong samples per step: {}\n".format(self.maxWrongSamplesPerStep)
            self.description += "   Inputs, size: {}\n".format(self.input, self.inputSize)
            self.description += "   Outputs, size: {}\n".format(self.output, self.outputSize)
        return self.description
