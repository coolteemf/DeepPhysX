import numpy as np

from DeepPhysX_Core.Environment.BaseEnvironment import BaseEnvironment, BytesNumpyConverter


class Environment2(BaseEnvironment):

    def __init__(self, ip_address='localhost', port=10000, data_converter=BytesNumpyConverter, instance_id=1):
        super(Environment2, self).__init__(ip_address=ip_address, port=port, data_converter=data_converter,
                                          instance_id=instance_id)
        self.tensor = np.random.random((3, 2))

    def create(self):
        print(f"Created Env n°{self.instance_id}")
        self.input_size = self.tensor.shape
        self.output_size = self.tensor.shape

    def onStep(self):
        self.tensor = np.random.random(self.input_size)
        self.send_training_data(network_input=self.tensor, network_output=2.0 * self.tensor)

    def close(self):
        print(f"Closing Env n°{self.instance_id}")

    def __str__(self):
        return f"Environment n°{self.instance_id} with tensor {self.tensor}"
