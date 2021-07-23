from DeepPhysX_Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from .SofaEnvironment import SofaEnvironment
from dataclasses import dataclass

import Sofa.Core


class SofaEnvironmentConfig(BaseEnvironmentConfig):
    @dataclass
    class SofaEnvironmentProperties(BaseEnvironmentConfig.BaseEnvironmentProperties):
        pass

    def __init__(self, environment_class=SofaEnvironment, simulations_per_step=1, max_wrong_samples_per_step=10,
                 always_create_data=False, visualizer_class=None,
                 multiprocessing=1, multiprocess_method=None, root_node=None):
        BaseEnvironmentConfig.__init__(self, environment_class=environment_class,
                                       simulations_per_step=simulations_per_step,
                                       max_wrong_samples_per_step=max_wrong_samples_per_step,
                                       always_create_data=always_create_data,
                                       visualizer_class=visualizer_class,
                                       number_of_thread=multiprocessing,
                                       multiprocess_method=multiprocess_method)
        self.rootNode = root_node
        self.environment_config = self.SofaEnvironmentProperties(simulations_per_step=simulations_per_step,
                                                                 max_wrong_samples_per_step=max_wrong_samples_per_step)
        self.descriptionName = "SOFA EnvironmentConfig"

    def setRootNodes(self):
        if self.number_of_thread == 1:
            self.rootNode = Sofa.Core.Node('rootNode')
        else:
            self.rootNode = [Sofa.Core.Node('rootNode' + str(i)) for i in range(self.number_of_thread)]

    def initNodes(self):
        if self.number_of_thread == 1:
            Sofa.Simulation.init(self.rootNode)
        else:
            for node in self.rootNode:
                Sofa.Simulation.init(node)

    def createEnvironment(self, training=True):
        if self.rootNode is None:
            self.setRootNodes()
        self.addRequiredPlugins()
        if self.number_of_thread == 1:
            environment = self.rootNode.addObject(self.environment_class(root_node=self.rootNode,
                                                                         config=self.environment_config,
                                                                         idx_instance=0))
            environment.create()
        else:
            environment = [self.rootNode[i].addObject(self.environment_class(root_node=self.rootNode[i],
                                                                             config=self.environment_config,
                                                                             idx_instance=i + 1))
                           for i in range(len(self.rootNode))]
        self.initNodes()
        return environment

    def addRequiredPlugins(self):
        pass
