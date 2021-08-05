from DeepPhysX_Core.Environment.BaseEnvironment import BaseEnvironment

import Sofa.Simulation


class SofaEnvironment(Sofa.Core.Controller, BaseEnvironment):

    def __init__(self, root_node, config, idx_instance=1, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.root = root_node
        BaseEnvironment.__init__(self, config=config, instance_id=idx_instance)
        self.description_name = "SOFA Environment"

    def create(self):
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

    def initVisualizer(self):
        pass

    def __str__(self):
        """
        :return: String containing information about the BaseEnvironmentConfig object
        """
        description = BaseEnvironment.__str__(self)
        return description