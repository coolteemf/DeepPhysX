from .Environment import Environment


class EnvironmentConfig:

    def __init__(self, simulation_per_step=1, always_create_data=False):

        self.environment = Environment
        self.environmentConfig = simulation_per_step
        self.alwaysCreateData = always_create_data

    def createEnvironment(self):
        return self.environment(self.environmentConfig)
