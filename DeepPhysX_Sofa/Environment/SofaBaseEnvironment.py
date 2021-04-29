from DeepPhysX.Environment.BaseEnvironment import BaseEnvironment

import Sofa.Simulation


class SofaBaseEnvironment(Sofa.Core.Controller, BaseEnvironment):

    def __init__(self, root_node, config, idx_instance=1,
                 *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.rootNode = root_node
        BaseEnvironment.__init__(self, config, idx_instance)
        self.descriptionName = "SOFA Environment"

    def create(self, config):
        print("[SOFA Environment] You have to implement environment create() method.")

    def reset(self):
        Sofa.Simulation.reset(self.rootNode)
        self.onReset()

    def onReset(self):
        pass

    def step(self):
        Sofa.Simulation.animate(self.rootNode, self.rootNode.dt.value)
        self.onStep()

    def onStep(self):
        pass

    def checkSample(self):
        pass
