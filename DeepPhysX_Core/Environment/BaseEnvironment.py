import numpy as np


class BaseEnvironment:

    def __init__(self, config, idx_instance=1, visualizer_class=None):

        self.name = "Environment n°{}".format(idx_instance)
        self.simulations_per_step = config.simulations_per_step
        self.max_wrong_samples_per_step = config.max_wrong_samples_per_step
        self.visualizer = visualizer_class() if visualizer_class is not None else None

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

    def initVisualizer(self):
        pass

    def renderVisualizer(self):
        if self.visualizer is not None:
            self.visualizer.render()

    def save_wrong_sample(self, session_dir):
        if self.visualizer is not None:
            self.visualizer.saveSample(session_dir)

    def getDescription(self):
        if len(self.description) == 0:
            self.description += "\n{}\n".format(self.description_name)
            self.description += "   Name: {}\n".format(self.name)
            self.description += "   Simulations per step: {}\n".format(self.simulations_per_step)
            self.description += "   Max wrong samples per step: {}\n".format(self.max_wrong_samples_per_step)
            self.description += "   Inputs, size: {}\n".format(self.input, self.input_size)
            self.description += "   Outputs, size: {}\n".format(self.output, self.output_size)
        return self.description
