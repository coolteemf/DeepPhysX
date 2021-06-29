from DeepPhysX_Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from .SofaEnvironment import SofaEnvironment
from dataclasses import dataclass

import Sofa.Core


class SofaEnvironmentConfig(BaseEnvironmentConfig):

    @dataclass
    class SofaEnvironmentProperties(BaseEnvironmentConfig.BaseEnvironmentProperties):
        pass

    def __init__(self, environment_class=SofaEnvironment, simulations_per_step=1, max_wrong_samples_per_step=10,
                 always_create_data=False, multiprocessing=1, multiprocess_method=None, root_node=None):
        BaseEnvironmentConfig.__init__(self, environment_class, simulations_per_step, max_wrong_samples_per_step,
                                       always_create_data, multiprocessing, multiprocess_method)
        self.rootNode = root_node
        self.environment_config = self.SofaEnvironmentProperties(simulations_per_step=simulations_per_step,
                                                                 max_wrong_samples_per_step=max_wrong_samples_per_step)
        self.descriptionName = "SOFA EnvironmentConfig"

    def setRootNodes(self):
        if self.multiprocessing == 1:
            self.rootNode = Sofa.Core.Node('rootNode')
        else:
            self.rootNode = [Sofa.Core.Node('rootNode'+str(i)) for i in range(self.multiprocessing)]

    def initNodes(self):
        if self.multiprocessing == 1:
            Sofa.Simulation.init(self.rootNode)
        else:
            for node in self.rootNode:
                Sofa.Simulation.init(node)

    def createEnvironment(self, training=True):
        if self.rootNode is None:
            self.setRootNodes()
        self.addRequiredPlugins()
        if self.multiprocessing == 1:
            environment = self.rootNode.addObject(self.environment_class(self.rootNode, self.environment_config, 0))
        else:
            environment = [self.rootNode[i].addObject(self.environment_class(self.rootNode[i], self.environment_config,
                                                                             i + 1)) for i in range(len(self.rootNode))]
        self.initNodes()
        return environment

    def addRequiredPlugins(self):
        pass

