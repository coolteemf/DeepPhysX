import numpy as np

from DeepPhysX_Sofa.Environment.SofaEnvironment import SofaEnvironment, BytesNumpyConverter


class EnvironmentSofa(SofaEnvironment):

    def __init__(self, root_node, ip_address='localhost', port=10000, data_converter=BytesNumpyConverter,
                 instance_id=1):
        super(EnvironmentSofa, self).__init__(ip_address=ip_address, port=port, data_converter=data_converter,
                                              instance_id=instance_id, root_node=root_node)
        self.tensor = np.random.random((3,1))

    def create(self):
        print(f"Created Env n°{self.instance_id}")
        self.input_size = self.tensor.shape
        self.output_size = self.tensor.shape

    def onSimulationInitDoneEvent(self, event):
        # Todo: Warning, the init is called in launcher.py after b'size' request in Client. Should be called just after
        # Todo: the create() method (use decorator in SofaEnvironment ?)
        self.input_size = self.tensor.shape
        self.output_size = self.tensor.shape

    def step(self):
        self.tensor = np.random.random(self.input_size)

    def computeInput(self):
        self.input = np.copy(self.tensor)

    def computeOutput(self):
        self.output = np.copy(self.tensor)

    def applyPrediction(self, prediction):
        pass

    def close(self):
        print(f"Closing Env n°{self.instance_id}")

    def __str__(self):
        return f"Environment n°{self.instance_id} with tensor {self.tensor}"
