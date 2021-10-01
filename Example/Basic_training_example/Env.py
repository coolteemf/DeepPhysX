from DeepPhysX_Core.Environment.BaseEnvironment import BaseEnvironment

from numpy import mean, pi, array
from numpy.random import random


class MeanEnvironment(BaseEnvironment):

    def __init__(self, ip_address='localhost', port=10000, data_converter=None, instance_id=1,
                 number_of_instances=1, visualizer_class=None, *args, **kwargs):
        BaseEnvironment.__init__(self, ip_address=ip_address, port=port, data_converter=data_converter,
                                 instance_id=instance_id, number_of_instances=number_of_instances, visualizer_class=visualizer_class)

    def send_visualization(self):
        return {}

    def create(self):
        print(f"Created client n°{self.instance_id}")
        # Get the data sizes
        self.input_size = [50]
        self.output_size = [1]

    def init(self):
        pass

    async def animate(self):
        self.input = pi * random(50)
        self.output = array([mean(self.input)])

    async def step(self):
        await self.animate()
        await self.onStep()

    async def onStep(self):

        await self.send_training_data(network_input=self.input, network_output=self.output)
        await self.send_command_done()

    def checkSample(self, check_input=True, check_output=True):
        return True

    def applyPrediction(self, prediction):
        pass

    def initVisualizer(self):
        pass

    def __str__(self):
        description = BaseEnvironment.__str__(self)
        return description
