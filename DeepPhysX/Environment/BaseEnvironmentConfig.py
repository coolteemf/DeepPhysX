import os
from .BaseEnvironment import BaseEnvironment
from dataclasses import dataclass


class BaseEnvironmentConfig:

    @dataclass
    class BaseEnvironmentProperties:
        simulations_per_step: int
        max_wrong_samples_per_step: int

    def __init__(self, environment_class=BaseEnvironment, simulations_per_step=1, max_wrong_samples_per_step=10,
                 always_create_data=False, multiprocessing=1, multiprocess_method=None):
        # Environment configuration
        self.environmentConfig = self.BaseEnvironmentProperties(simulations_per_step=simulations_per_step,
                                                                max_wrong_samples_per_step=max_wrong_samples_per_step)
        # Environment variables
        self.environment_class = environment_class
        self.environmentClassName = environment_class.__name__
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
        self.descriptionName = "CORE EnvironmentConfig"
        self.description = ""

    def createEnvironment(self):
        if self.multiprocessing == 1:
            return self.environment_class(self.environmentConfig)
        return [self.environment_class(self.environmentConfig, i + 1) for i in range(self.multiprocessing)]

    def getDescription(self):
        if len(self.description) == 0:
            self.description += "\n{}\n".format(self.descriptionName)
            self.description += "   (environment) Environment class: {}\n".format(self.environment_class.__name__)
            self.description += "   (environment) Environment config: {}\n".format(self.environmentConfig)
            self.description += "   (environmentManager) Always create data: {}\n".format(self.alwaysCreateData)
            self.description += "   (environmentManager) Multiprocessing value: {}\n".format(self.multiprocessing)
        return self.description
