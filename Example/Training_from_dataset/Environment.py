import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
from numpy import array

from DeepPhysX_Core.Environment.BaseEnvironment import BaseEnvironment


class Environment(BaseEnvironment):

    def __init__(self, instance_id=1, number_of_instances=1, as_tcp_ip_client=True, ip_address='localhost', port=10000,
                 environment_manager=None):

        BaseEnvironment.__init__(self, instance_id=instance_id, number_of_instances=number_of_instances,
                                 as_tcp_ip_client=as_tcp_ip_client, ip_address=ip_address, port=port,
                                 environment_manager=environment_manager)

        self.input = array([[0.], [0.]])
        self.input_size = self.input.shape
        self.output = array([[0.], [0.]])
        self.output_size = self.output.shape

        self.nb_step = 0

    def create(self):
        pass

    async def step(self):
        if self.sample_in is None or self.sample_out is None:
            # Compute network in / out data
            self.input = array([[self.nb_step], [self.nb_step]])
            self.output = array([[2 * self.nb_step], [2 * self.nb_step]])
            # Setting (and sending) training data
            # TODO: if inverted, last additional sample is not sent
            self.set_additional_in_dataset(label="custom", data=self.input)
            self.set_additional_out_dataset(label="custom", data=self.output)
            self.set_training_data(input_array=self.input, output_array=self.output)
            # Increment iteration counter
            self.nb_step += 1
        else:
            # Add 0.5 to existing dataset
            self.input = array([self.sample_in[0], self.sample_in[1] + 0.5])
            self.output = array([self.sample_out[0], self.sample_out[1] + 0.5])
            self.set_training_data(input_array=self.input, output_array=self.output)
