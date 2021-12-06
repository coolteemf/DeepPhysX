from DeepPhysX_Core.Environment.BaseEnvironment import BaseEnvironment

from numpy import mean, pi, array
from numpy.random import random

import time

# This class generate a random vector between 0 and pi of 50 value and compute its mean.
# The vector is the input of the network and the groundtruth is the mean.
# The data are generated on the animate function
# The data are sent to the artificial neural network in the onStep function

class MeanEnvironment(BaseEnvironment):

    def __init__(self,
                 ip_address='localhost',
                 port=10000,
                 data_converter=None,
                 instance_id=1,
                 number_of_instances=1,
                 visualizer_class=None):
        BaseEnvironment.__init__(self,
                                 ip_address=ip_address,
                                 port=port,
                                 data_converter=data_converter,
                                 instance_id=instance_id,
                                 number_of_instances=number_of_instances,
                                 visualizer_class=visualizer_class)

    def create(self):
        print(f"Created client n°{self.instance_id}")

    async def animate(self):
        self.input = pi * random((25, 2))
        self.output = mean(self.input, axis=0)

    async def step(self):
        await self.animate()
        await self.onStep()

    async def onStep(self):
        await self.send_training_data(network_input=self.input, network_output=self.output)
