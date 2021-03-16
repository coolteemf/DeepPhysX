import numpy as np

from DeepPhysX.Environment.EnvironmentConfig import EnvironmentConfig


class EnvironmentManager:

    def __init__(self, environment_config: EnvironmentConfig):
        self.environmentConfig = environment_config
        self.environment = environment_config.createEnvironment()

    def step(self):
        for _ in range(self.environment.simulationPerStep):
            self.environment.step()

    def getData(self, batch_size, get_inputs, get_outputs):
        inputs = np.empty((batch_size, self.environment.inputSize))
        outputs = np.empty((batch_size, self.environment.outputSize))
        for i in range(batch_size):
            self.step()
            if get_inputs:
                inputs[i] = self.environment.getInput()
            if get_outputs:
                outputs[i] = self.environment.getOutput()
        return {'in': inputs, 'out': outputs}

    def close(self):
        pass
