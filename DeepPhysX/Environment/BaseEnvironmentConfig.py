import os
from DeepPhysX.Environment.BaseEnvironment import BaseEnvironment
from dataclasses import dataclass


class BaseEnvironmentConfig:

    @dataclass
    class BaseEnvironmentProperties:
        simulations_per_step: int
        max_wrong_samples_per_step: int

    def __init__(self, environment_class=BaseEnvironment, simulations_per_step=1, max_wrong_samples_per_step=10,
                 always_create_data=False, multiprocessing=1, multiprocess_method=None):

        # Check the arguments before to configure anything
        if type(simulations_per_step) != int and simulations_per_step < 1:
            raise TypeError("[BASEENVIRONMENTCONFIG] The number of simulations per step must be an int > 1.")
        if type(max_wrong_samples_per_step) != int and max_wrong_samples_per_step < 0:
            raise TypeError("[BASEENVIRONMENTCONFIG] The number of max wrong samples per step must be a positive int.")
        if type(always_create_data) != bool:
            raise TypeError("[BASEENVIRONMENTCONFIG] Always create data must be a boolean.")
        if type(multiprocessing) != int and multiprocessing < 0:
            raise TypeError("[BASEENVIRONMENTCONFIG] The multiprocessing number must be a positive int.")
        if multiprocess_method is not None and multiprocess_method not in ['process', 'pool']:
            raise ValueError("[BASEENVIRONMENTCONFIG] The multiprocessing method must be either process or pool.")

        # Environment configuration
        self.environment_config = self.BaseEnvironmentProperties(simulations_per_step=simulations_per_step,
                                                                 max_wrong_samples_per_step=max_wrong_samples_per_step)
        # Environment variables
        self.environment_class = environment_class
        # EnvironmentManager variables
        self.always_create_data = always_create_data
        self.multiprocessing = min(max(multiprocessing, 1), os.cpu_count())  # Assert nb is between 1 and cpu_count
        if self.multiprocessing > 1:
            if multiprocess_method is not None:
                self.multiprocess_method = multiprocess_method
            else:
                self.multiprocess_method = 'process'
                print("[BASEENVIRONMENTCONFIG]: The default multiprocessing method is set to 'process'.")
        else:
            self.multiprocess_method = multiprocess_method
        # Description
        self.descriptionName = "CORE EnvironmentConfig"
        self.description = ""

    def createEnvironment(self):
        if self.multiprocessing == 1:
            try:
                environment = self.environment_class(self.environment_config)
            except:
                raise TypeError("[BASEENVIRONMENTCONFIG] The given environment class is not a BaseEnvironment class.")
            if not isinstance(environment, BaseEnvironment):
                raise TypeError("[BASEENVIRONMENTCONFIG] The environment class must be a BaseEnvironment class.")
            return environment
        else:
            try:
                environments = [self.environment_class(self.environment_config,
                                                       i + 1) for i in range(self.multiprocessing)]
            except:
                raise TypeError("[BASEENVIRONMENTCONFIG] The given environment class is not a BaseEnvironment class.")
            if not isinstance(environments[0], BaseEnvironment):
                raise TypeError("[BASEENVIRONMENTCONFIG] The environment class must be a BaseEnvironment class.")
            return environments

    def getDescription(self):
        if len(self.description) == 0:
            self.description += "\n{}\n".format(self.descriptionName)
            self.description += "   (environment) Environment class: {}\n".format(self.environment_class.__name__)
            self.description += "   (environment) Environment config: {}\n".format(self.environment_config)
            self.description += "   (environmentManager) Always create data: {}\n".format(self.always_create_data)
            self.description += "   (environmentManager) Multiprocessing value: {}\n".format(self.multiprocessing)
        return self.description
