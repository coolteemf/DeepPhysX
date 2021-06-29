import os
from DeepPhysX_Core.Environment.BaseEnvironment import BaseEnvironment
from dataclasses import dataclass


class BaseEnvironmentConfig:
    @dataclass
    class BaseEnvironmentProperties:
        simulations_per_step: int
        max_wrong_samples_per_step: int

    def __init__(self, environment_class=BaseEnvironment, simulations_per_step=1, max_wrong_samples_per_step=10,
                 always_create_data=False, multiprocessing=1, multiprocess_method=None):

        # Description
        self.name = self.__class__.__name__
        self.description = ""

        # Check the arguments before to configure anything
        if type(simulations_per_step) != int and simulations_per_step < 1:
            raise TypeError("[{}] The number of simulations per step must be an int > 1.".format(self.name))
        if type(max_wrong_samples_per_step) != int and max_wrong_samples_per_step < 0:
            raise TypeError("[{}] The number of max wrong samples per step must be a positive int.".format(self.name))
        if type(always_create_data) != bool:
            raise TypeError("[{}] Always create data must be a boolean.".format(self.name))
        if type(multiprocessing) != int and multiprocessing < 0:
            raise TypeError("[{}] The multiprocessing number must be a positive int.".format(self.name))
        if multiprocess_method is not None and multiprocess_method not in ['process', 'pool']:
            raise ValueError("[{}] The multiprocessing method must be either process or pool.".format(self.name))

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
                print("[{}]: The default multiprocessing method is set to 'process'.".format(self.name))
        else:
            self.multiprocess_method = multiprocess_method

    def createEnvironment(self):
        if self.multiprocessing == 1:
            try:
                environment = self.environment_class(self.environment_config)
            except:
                raise TypeError("[{}] The given environment class is not a BaseEnvironment class.".format(self.name))
            if not isinstance(environment, BaseEnvironment):
                raise TypeError("[{}] The environment class must be a BaseEnvironment class.".format(self.name))
            return environment
        else:
            try:
                environments = [self.environment_class(self.environment_config,
                                                       i + 1) for i in range(self.multiprocessing)]
            except:
                raise TypeError("[{}] The given environment class is not a BaseEnvironment class.".format(self.name))
            if not isinstance(environments[0], BaseEnvironment):
                raise TypeError("[{}] The environment class must be a BaseEnvironment class.".format(self.name))
            return environments

    def getDescription(self):
        if len(self.description) == 0:
            self.description += "\n{}\n".format(self.name)
            self.description += "   (environment) Environment class: {}\n".format(self.environment_class.__name__)
            self.description += "   (environment) Environment config: {}\n".format(self.environment_config)
            self.description += "   (environmentManager) Always create data: {}\n".format(self.always_create_data)
            self.description += "   (environmentManager) Multiprocessing value: {}\n".format(self.multiprocessing)
        return self.description
