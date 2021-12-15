import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np

from DeepPhysX_Core.Environment.BaseEnvironment import BaseEnvironment


class Environment(BaseEnvironment):

    def __init__(self,
                 instance_id=1,
                 number_of_instances=1,
                 as_tcpip_client=True,
                 ip_address='localhost',
                 port=10000,
                 environment_manager=None):

        BaseEnvironment.__init__(self,
                                 instance_id=instance_id,
                                 number_of_instances=number_of_instances,
                                 as_tcpip_client=as_tcpip_client,
                                 ip_address=ip_address,
                                 port=port,
                                 environment_manager=environment_manager)
        self.input = np.array([0., 0.])
        self.output = np.array([0., 0.])
        # Important to define the data sizes
        self.input_size = self.input.shape
        self.output_size = self.output.shape

        self.nb_iterations = 0

    def create(self):
        pass

    async def step(self):
        # First epoch case: produce data
        if self.sample_in is None and self.sample_out is None:
            # Compute network in / out data
            self.input = np.array([100. * self.nb_iterations, 100. * self.nb_iterations])
            self.output = np.array([200. * self.nb_iterations, 200. * self.nb_iterations])
            # Compute additional data to be stored in dataset
            duplicated_input = self.input.copy()
            duplicated_output = self.output.copy()
            # Additional data must be set before setting training data
            self.additionalInDataset(label="duplicated", data_array=duplicated_input)
            self.additionalOutDataset(label="duplicated", data_array=duplicated_output)
            # Setting (and sending) training data
            self.setTrainingData(input_array=self.input, output_array=self.output)
            # Increment iteration counter
            self.nb_iterations += 1
        # Other epochs case: data is sent to the Environment if use_dataset_in_environment flag for Config is True
        else:
            print("Input sample")
            print(self.sample_in, self.additional_inputs)
            print("Output sample")
            print(self.sample_out, self.additional_outputs)


