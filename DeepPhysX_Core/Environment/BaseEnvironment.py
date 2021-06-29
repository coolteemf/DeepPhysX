import numpy as np


class BaseEnvironment:

    def __init__(self, config, idx_instance=1):

        self.name = "Environment n°{}".format(idx_instance)
        self.simulations_per_step = config.simulations_per_step
        self.max_wrong_samples_per_step = config.max_wrong_samples_per_step

        self.input, self.output = np.array([]), np.array([])
        self.input_size, self.output_size = None, None

        self.description = ""
        self.description_name = self.__class__.__name__

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
            self.description += "\n{}\n".format(self.description_name)
            self.description += "   Name: {}\n".format(self.name)
            self.description += "   Simulations per step: {}\n".format(self.simulations_per_step)
            self.description += "   Max wrong samples per step: {}\n".format(self.max_wrong_samples_per_step)
            self.description += "   Inputs, size: {}\n".format(self.input, self.input_size)
            self.description += "   Outputs, size: {}\n".format(self.output, self.output_size)
        return self.description
