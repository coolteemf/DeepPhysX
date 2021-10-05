from DeepPhysX_Core import BaseEnvironment

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
                                 visual_object=visualizer_class)

    def send_visualization(self):
        # There is no visualisation, we send an empty dictionnary
        return {}

    def create(self):
        print(f"Created client n°{self.instance_id}")
        # Data size HAVE to be set before we start sending data
        # Get the data sizes
        self.input_size = [25, 2]
        self.output_size = [2]

    async def animate(self):
        self.input = pi * random((25, 2))
        self.output = array([mean(self.input, axis=0)])
        time.sleep(0.001)

    async def step(self):
        await self.animate()
        await self.onStep()

    async def onStep(self):
        await self.send_training_data(network_input=self.input, network_output=self.output)
        await self.send_command_done()
        #print("DATA SENT")

    def checkSample(self, check_input=True, check_output=True):
        return True
