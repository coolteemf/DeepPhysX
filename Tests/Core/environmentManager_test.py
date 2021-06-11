from MyEnvironment import MyBaseEnvironment
from DeepPhysX.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX.Manager.EnvironmentManager import EnvironmentManager


def main():

    # Single environment management
    single_environment_config = BaseEnvironmentConfig(environment_class=MyBaseEnvironment,
                                                      simulations_per_step=1,
                                                      always_create_data=True,
                                                      multiprocessing=1,
                                                      multiprocess_method=None)
    single_environment_manager = EnvironmentManager(environment_config=single_environment_config)
    for _ in range(10):
        single_environment_manager.step()
    print("")
    single_environment_manager.environment.reset()
    single_environment_manager.environment.simulations_per_step = 3
    for _ in range(2):
        data = single_environment_manager.getData(batch_size=3, get_inputs=True, get_outputs=False)
        print("Data steps for {} : {}".format(single_environment_manager.environment.name, list(data['in'])))

    # Multiple environment management
    print("")
    multiple_environment_config = BaseEnvironmentConfig(environment_class=MyBaseEnvironment,
                                                        simulations_per_step=1,
                                                        always_create_data=True,
                                                        multiprocessing=5,
                                                        multiprocess_method='process')
    multiple_environment_manager = EnvironmentManager(environment_config=multiple_environment_config)
    for _ in range(2):
        data = multiple_environment_manager.getData(batch_size=12, get_inputs=True, get_outputs=True)
        for env in multiple_environment_manager.environment:
            print("Current step {} : {}".format(env.name, env.getInput()))

    return


if __name__ == '__main__':
    main()
