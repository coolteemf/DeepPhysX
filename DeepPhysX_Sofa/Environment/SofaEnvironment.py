from DeepPhysX_Core.Environment.BaseEnvironment import BaseEnvironment, BytesNumpyConverter

import Sofa
import Sofa.Simulation


class SofaEnvironment(Sofa.Core.Controller, BaseEnvironment):

    def __init__(self, root_node, ip_address='localhost', port=10000, data_converter=BytesNumpyConverter, instance_id=1,
                 *args, **kwargs):

        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.root = root_node
        BaseEnvironment.__init__(self, ip_address=ip_address, port=port, data_converter=data_converter,
                                 instance_id=instance_id)

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
