import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
from numpy import array

from DeepPhysX_Core.Environment.BaseEnvironment import BaseEnvironment


class Environment(BaseEnvironment):

    def __init__(self, instance_id=1, number_of_instances=1, as_tcpip_client=True, ip_address='localhost', port=10000,
                 visual_object=None, environment_manager=None):

        BaseEnvironment.__init__(self, instance_id=instance_id, number_of_instances=number_of_instances,
                                 as_tcpip_client=as_tcpip_client, ip_address=ip_address, port=port,
                                 visual_object=visual_object, environment_manager=environment_manager)

        self.input = array([0., 0.])
        self.input_size = self.input.shape
        self.output = array([0., 0.])
        self.output_size = self.output.shape

        self.nb_step = 0

    def create(self):
        pass

    async def step(self):
        # Compute network in / out data
        self.input = array([self.nb_step, self.nb_step])
        self.output = array([2 * self.nb_step, 2 * self.nb_step])
        # Setting (and sending) training data
        self.setTrainingData(input_array=self.input, output_array=self.output)
        # Increment iteration counter
        self.nb_step += 1
