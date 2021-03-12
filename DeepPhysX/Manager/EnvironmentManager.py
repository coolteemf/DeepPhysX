import numpy as np

from DeepPhysX.Environment.EnvironmentConfig import EnvironmentConfig


class EnvironmentManager:

    def __init__(self, environment_config: EnvironmentConfig):
        self.environmentConfig = environment_config
        self.environment = environment_config.createEnvironment()

    def step(self):
        for _ in range(self.environment.simulationPerStep):
            self.environment.step()

    def getData(self, get_inputs, get_outputs, batch_size):
        # Todo : add get_inputs and get_outputs booleans
        inputs = np.empty((batch_size, self.environment.inputSize))
        outputs = np.empty((batch_size, self.environment.outputSize))
        for i in range(batch_size):
            self.step()
            inputs[i] = self.environment.getInput()
            outputs[i] = self.environment.getOutput()
        return {'in': inputs, 'out': outputs}
