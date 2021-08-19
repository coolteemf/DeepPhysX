import numpy as np

from DeepPhysX_Sofa.Environment.SofaEnvironment import SofaEnvironment, BytesNumpyConverter


class EnvironmentSofa(SofaEnvironment):

    def __init__(self, root_node, ip_address='localhost', port=10000, data_converter=BytesNumpyConverter,
                 instance_id=1):
        super(EnvironmentSofa, self).__init__(ip_address=ip_address, port=port, data_converter=data_converter,
                                              instance_id=instance_id, root_node=root_node)

    def create(self):
        print(f"Created Env n°{self.instance_id}")

    def onSimulationInitDoneEvent(self, value):
        self.input_size = (50, 50, 3)
        self.output_size = (50, 50, 3)

    def onAnimateBeginEvent(self, value):
        self.input = np.random.random(self.input_size)
        self.output = 0.1 * np.random.random(self.output_size)

    def onAnimateEndEvent(self, value):
        # if self.compute_essential_data:
        #     self.sync_send_training_data(network_input=self.input, network_output=self.output)
        # self.sync_send_command_done()
        return

    async def onStep(self):
        if self.compute_essential_data:
            await self.send_training_data(network_input=self.input, network_output=self.output)
        await self.send_command_done()

    def applyPrediction(self, prediction):
        pass

    def close(self):
        print(f"Closing Env n°{self.instance_id}")

    def __str__(self):
        return f"Environment n°{self.instance_id} with tensor {self.tensor}"
