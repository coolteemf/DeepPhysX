import os
from DeepPhysX_Core.Environment.BaseEnvironment import BaseEnvironment
from DeepPhysX_Core.Visualizer.VedoVisualizer import VedoVisualizer
from dataclasses import dataclass


class BaseEnvironmentConfig:
    @dataclass
    class BaseEnvironmentProperties:
        """
        Class containing data to create Environment object.
        """
        simulations_per_step: int
        max_wrong_samples_per_step: int

    def __init__(self, environment_class=BaseEnvironment, simulations_per_step=1, max_wrong_samples_per_step=10,
                 always_create_data=False, record_wrong_samples=False,
                 number_of_thread=1, multiprocess_method=None):
        """
        BaseEnvironmentConfig is a configuration class to parameterize and create a BaseEnvironment for the
        EnvironmentManager.

        :param environment_class: Class from which an instance will be created
        :type environment_class: type[BaseEnvironment]
        :param int simulations_per_step: Number of iterations to compute in the Environment at each time step
        :param int max_wrong_samples_per_step: Maximum number of wrong samples to produce in a step
        :param bool always_create_data: If True, data will always be created from environment. If False, data will be
                                        created from the environment during the first epoch and then re-used from the
                                        Dataset.
        :param bool record_wrong_samples: If True, wrong samples are recorded through Visualizer
        :param int number_of_thread: Number of thread to run
        :param multiprocess_method: Values at \'process\' or \'pool\'
        """

        self.name = self.__class__.__name__

        # Check simulations_per_step type and value
        if type(simulations_per_step) != int:
            raise TypeError(f"[{self.name}] Wrong simulations_per_step type: int required, get "
                            f"{type(simulations_per_step)}")
        if simulations_per_step < 1:
            raise ValueError(f"[{self.name}] Given simulations_per_step value is negative or null")
        # Check max_wrong_samples_per_step type and value
        if type(max_wrong_samples_per_step) != int:
            raise TypeError(f"[{self.name}] Wrong max_wrong_samples_per_step type: int required, get "
                            f"{type(max_wrong_samples_per_step)}")
        if simulations_per_step < 1:
            raise ValueError(f"[{self.name}] Given max_wrong_simulations_per_step value is negative or null")
        # Check always_create_data type
        if type(always_create_data) != bool:
            raise TypeError(f"[{self.name}] Wrong always_create_data type: bool required, get "
                            f"{type(always_create_data)}")

        # Todo : Multiprocessing in Environment package level ?
        if type(number_of_thread) != int and number_of_thread < 0:
            raise TypeError(f"[{self.name}] The number_of_thread number must be a positive int.")
        if multiprocess_method is not None and multiprocess_method not in ['process', 'pool']:
            raise ValueError(f"[{self.name}] The multiprocessing method must be either process or pool.")

        # BaseEnvironment parameterization
        self.environment_class = environment_class
        self.environment_config = self.BaseEnvironmentProperties(simulations_per_step=simulations_per_step,
                                                                 max_wrong_samples_per_step=max_wrong_samples_per_step)

        # EnvironmentManager parameterization
        self.always_create_data = always_create_data
        self.record_wrong_samples = record_wrong_samples

        # Todo : Multiprocessing in Environment package level ?
        self.number_of_thread = min(max(number_of_thread, 1), os.cpu_count())  # Assert nb is between 1 and cpu_count
        if self.number_of_thread > 1:
            if multiprocess_method is not None:
                self.multiprocess_method = multiprocess_method
            else:
                self.multiprocess_method = 'process'
                print(f"[{self.name}]: The default multiprocessing method is set to 'process'.")
        else:
            self.multiprocess_method = multiprocess_method

    def createEnvironment(self, environment_manager=None):
        """
        :return: BaseEnvironment object from environment_class and its parameters
        """
        if self.number_of_thread == 1:
            # Create environment
            try:
                environment = self.environment_class(config=self.environment_config)
                environment.environment_manager = environment_manager
            except:
                raise ValueError(f"[{self.name}] Given environment_class got an unexpected keyword argument 'config'")
            if not isinstance(environment, BaseEnvironment):
                raise TypeError(f"[{self.name}] Wrong environment_class type: BaseEnvironment required, get "
                                f"{self.environment_class}")
            environment.create()

            return environment
        # Todo : Multiprocessing in Environment package level ?
        else:
            try:
                environments = [self.environment_class(config=self.environment_config,
                                                       idx_instance=i + 1) for i in range(self.number_of_thread)]
                for env in environments:
                    env.environment_manager = environment_manager
            except:
                raise TypeError("[{}] The given environment class is not a BaseEnvironment class.".format(self.name))
            if not isinstance(environments[0], BaseEnvironment):
                raise TypeError("[{}] The environment class must be a BaseEnvironment class.".format(self.name))
            return environments

    def __str__(self):
        """
        :return: String containing information about the BaseEnvironmentConfig object
        """
        description = "\n"
        description += f"{self.name}\n"
        description += f"    Environment class: {self.environment_class.__name__}\n"
        description += f"    Simulations per step: {self.environment_config.simulations_per_step}\n"
        description += f"    Max wrong samples per step: {self.environment_config.max_wrong_samples_per_step}\n"
        description += f"    Always create data: {self.always_create_data}\n"
        return description
