"""
Environment
At each step, generate a random vector of 50 values between [0, pi] and compute its mean.
The random vector is the input of the network and the ground truth is the mean.
"""

# Python related imports
from numpy import mean, pi
from numpy.random import random, randint
from time import sleep

# Session related imports
from Environment import MeanEnvironment


# Create an Environment as a BaseEnvironment child class
class MeanEnvironmentOffscreen(MeanEnvironment):

    def __init__(self,
                 ip_address='localhost',
                 port=10000,
                 instance_id=1,
                 number_of_instances=1,
                 as_tcp_ip_client=True,
                 environment_manager=None):

        MeanEnvironment.__init__(self,
                                 ip_address=ip_address,
                                 port=port,
                                 instance_id=instance_id,
                                 number_of_instances=number_of_instances,
                                 as_tcp_ip_client=as_tcp_ip_client,
                                 environment_manager=environment_manager)

    def send_visualization(self):
        # Return no visualization data
        return {}

    async def step(self):
        # Compute new data
        if not self.constant:
            self.input_value = pi * random(self.data_size)
            self.output_value = mean(self.input_value, axis=0)
        # Simulate longer process
        if self.sleep:
            sleep(0.01 * randint(0, 10))
        # Send the training data
        self.set_training_data(input_array=self.input_value,
                               output_array=self.output_value)

    def apply_prediction(self, prediction):
        # No update
        pass
