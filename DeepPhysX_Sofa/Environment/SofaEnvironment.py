from DeepPhysX.Environment.Environment import Environment

import Sofa.Simulation


class SofaEnvironment(Sofa.Core.Controller, Environment):

    def __init__(self, root_node, simulations_per_step=1, max_wrong_samples=10, idx_instance=1,
                 *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.rootNode = root_node
        Environment.__init__(self, simulations_per_step, max_wrong_samples, idx_instance)
        self.descriptionName = "SOFA Environment"

    def create(self):
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
