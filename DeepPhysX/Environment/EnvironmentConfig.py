import os


class EnvironmentConfig:

    def __init__(self, environment_class, simulations_per_step=1, max_wrong_samples_per_step=10,
                 always_create_data=False, multiprocessing=1, multiprocess_method=None):
        # Environment variables
        self.environment_class = environment_class
        self.environmentConfig = simulations_per_step, max_wrong_samples_per_step
        # EnvironmentManager variables
        self.alwaysCreateData = always_create_data
        self.multiprocessing = min(max(multiprocessing, 1), os.cpu_count())  # Assert nb is between 1 and cpu_count
        if self.multiprocessing > 1:
            if (multiprocess_method is not None) and (multiprocess_method in ['process', 'pool']):
                self.multiprocessMethod = multiprocess_method
            else:
                print("EnvironmentConfig: The chosen multiprocessing method must be 'process' (default) or 'pool'.")
                self.multiprocessMethod = 'process'
        else:
            self.multiprocessMethod = multiprocess_method
        # Description
        self.description = ""

    def createEnvironment(self):
        if self.multiprocessing == 1:
            return self.environment_class(*self.environmentConfig)
        return [self.environment_class(*self.environmentConfig, i+1) for i in range(self.multiprocessing)]

    def getDescription(self):
        if len(self.description) == 0:
            self.description += "\nCORE EnvironmentConfig:\n"
            self.description += "   (environment) Environment class: {}\n".format(self.environment_class.__name__)
            self.description += "   (environment) Environment config: {}\n".format(self.environmentConfig)
            self.description += "   (environmentManager) Always create data: {}\n".format(self.alwaysCreateData)
            self.description += "   (environmentManager) Multiprocessing value: {}\n".format(self.multiprocessing)
        return self.description
