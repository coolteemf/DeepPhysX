from DeepPhysX.Environment.BaseEnvironment import BaseEnvironment

import Sofa.Simulation


class SofaBaseEnvironment(Sofa.Core.Controller, BaseEnvironment):

    def __init__(self, root_node, config, training, idx_instance=1,
                 *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.root = root_node
        BaseEnvironment.__init__(self, config, training, idx_instance)
        self.descriptionName = "SOFA Environment"

    def create(self, config):
        raise NotImplementedError

    def step(self):
        Sofa.Simulation.animate(self.root, self.root.dt.value)
        self.onStep()

    def onStep(self):
        pass

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

    def applyPrediction(self, prediction):
        raise NotImplementedError
