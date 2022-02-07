import numpy as np

from DeepPhysX_Core.Environment.BaseEnvironment import BaseEnvironment


class TestEnvironment(BaseEnvironment):

    def __init__(self, ip_address='localhost', port=10000, instance_id=1, number_of_instances=1,
                 as_tcp_ip_client=True, environment_manager=None):
        BaseEnvironment.__init__(self, ip_address=ip_address, port=port, instance_id=instance_id,
                                 number_of_instances=number_of_instances, as_tcp_ip_client=as_tcp_ip_client,
                                 environment_manager=environment_manager)
        # Check init functions call
        self.call_create = False
        self.call_init = False
        # Parameters to receive
        self.parameters = {}
        # Environment parameters that must be learned by network
        self.p = [round(np.random.randn(), 2) for _ in range(4)]
        self.idx_step = 0

    def create(self):
        # Assert method is called
        self.call_create = True
        # Define sizes
        self.input_size = np.random.randn(1).size
        self.output_size = self.input_size

    def init(self):
        # Assert method is called
        self.call_init = True

    def recv_parameters(self, param_dict):
        self.parameters = param_dict

    def send_parameters(self):
        dict_to_return = {}
        for key, value in self.parameters.items():
            dict_to_return[key] = value * int(key[-1])
        return dict_to_return

    def step(self):
        self.idx_step += 1

    def computeInput(self):
        self.input = np.random.randn(1).round(2)

    def computeOutput(self):
        self.output = 0
        for i in range(len(self.p)):
            self.output += self.p[i] * (self.input ** i)

    def reset(self):
        self.idx_step = 0

    def apply_prediction(self, prediction):
        pass
