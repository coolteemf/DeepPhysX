import numpy as np
from dataclasses import dataclass

from DeepPhysX_Core.Environment.BaseEnvironment import BaseEnvironment
from DeepPhysX_Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig


class NumpyEnvironment(BaseEnvironment):

    def __init__(self, config):
        BaseEnvironment.__init__(self, config=config)
        self.idx_step = 0
        # Environment parameters that must be learned by network
        self.p = [round(np.random.randn(), 2) for _ in range(config.nb_parameters)]

    def create(self):
        self.input_size = np.random.randn(1).size
        self.output_size = self.input_size

    def step(self):
        self.idx_step += self.simulations_per_step

    def computeInput(self):
        self.input = np.random.randn(1).round(2)

    def computeOutput(self):
        self.output = 0
        for i in range(len(self.p)):
            self.output += self.p[i] * (self.input ** i)

    def reset(self):
        self.idx_step = 0


class NumpyEnvironmentConfig(BaseEnvironmentConfig):
    @dataclass
    class NumpyEnvironmentProperties(BaseEnvironmentConfig.BaseEnvironmentProperties):
        nb_parameters: int

    def __init__(self, environment_class=NumpyEnvironment, simulations_per_step=1, max_wrong_samples_per_step=10,
                 always_create_data=False, nb_parameters=4):
        super(NumpyEnvironmentConfig, self).__init__(environment_class=environment_class,
                                                     always_create_data=always_create_data)
        self.environment_config = self.NumpyEnvironmentProperties(simulations_per_step=simulations_per_step,
                                                                  max_wrong_samples_per_step=max_wrong_samples_per_step,
                                                                  nb_parameters=nb_parameters)
