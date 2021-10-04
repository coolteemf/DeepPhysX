from DeepPhysX_Core.Environment.BaseEnvironment import BaseEnvironment

import Sofa
import Sofa.Simulation


class SofaEnvironment(Sofa.Core.Controller, BaseEnvironment):

    def __init__(self,
                 root_node,
                 ip_address='localhost',
                 port=10000,
                 data_converter=None,
                 instance_id=1,
                 number_of_instances=1,
                 as_tcpip_client=True,
                 visual_object=None,
                 environment_manager=None,
                 *args, **kwargs):

        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.root = root_node
        BaseEnvironment.__init__(self, ip_address=ip_address, port=port, data_converter=data_converter,
                                 instance_id=instance_id, number_of_instances=number_of_instances,
                                 as_tcpip_client=as_tcpip_client, visual_object=visual_object,
                                 environment_manager=environment_manager)

    def create(self):
        raise NotImplementedError

    def init(self):
        Sofa.Simulation.init(self.root)

    async def animate(self):
        Sofa.Simulation.animate(self.root, self.root.dt.value)

    async def step(self):
        await self.animate()
        await self.onStep()

    async def onStep(self):
        pass

    def checkSample(self, check_input=True, check_output=True):
        return True

    def applyPrediction(self, prediction):
        pass

    def initVisualizer(self):
        pass

    def __str__(self):
        """
        :return: String containing information about the BaseEnvironmentConfig object
        """
        description = BaseEnvironment.__str__(self)
        return description
