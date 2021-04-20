from DeepPhysX.Environment.EnvironmentConfig import EnvironmentConfig

import SofaRuntime
import Sofa.Core


class SofaEnvironmentConfig(EnvironmentConfig):

    def __init__(self, environment_class, simulations_per_step=1, max_wrong_samples_per_step=10,
                 always_create_data=False, multiprocessing=1, multiprocess_method=None):
        EnvironmentConfig.__init__(self, environment_class, simulations_per_step, max_wrong_samples_per_step,
                                   always_create_data, multiprocessing, multiprocess_method)
        self.rootNode = None
        self.descriptionName = "SOFA EnvironmentConfig"

    def setRootNodes(self):
        if self.multiprocessing == 1:
            self.rootNode = Sofa.Core.Node('rootNode')
        else:
            self.rootNode = [Sofa.Core.Node('rootNode'+str(i)) for i in range(self.multiprocessing)]

    def createEnvironment(self):
        self.setRootNodes()
        if self.multiprocessing == 1:
            environment = self.rootNode.addObject(self.environment_class(self.rootNode, *self.environmentConfig, 0))
        else:
            environment = [self.rootNode[i].addObject(self.environment_class(self.rootNode[i], *self.environmentConfig,
                                                                             i+1)) for i in range(len(self.rootNode))]
        return environment

