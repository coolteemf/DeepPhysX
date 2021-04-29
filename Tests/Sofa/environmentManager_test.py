from MyEnvironment import MyEnvironment
from DeepPhysX_Sofa.Environment.SofaBaseEnvironmentConfig import SofaBaseEnvironmentConfig
from DeepPhysX.Manager.EnvironmentManager import EnvironmentManager


def main():

    # Single environment management
    single_environment_config = SofaBaseEnvironmentConfig(environment_class=MyEnvironment,
                                                          simulations_per_step=1,
                                                          always_create_data=True,
                                                          multiprocessing=1,
                                                          multiprocess_method=None)
    single_environment_manager = EnvironmentManager(environment_config=single_environment_config)
    print(single_environment_config.getDescription())
    print(single_environment_manager.environment.getDescription())
    # Single step
    for _ in range(10):
        single_environment_manager.step()
    print(single_environment_manager.environment.getInput())
    print("")
    # Multi steps
    single_environment_manager.environment.reset()
    single_environment_manager.environment.simulationsPerStep = 3
    for _ in range(10):
        single_environment_manager.step()
    print(single_environment_manager.environment.getInput())
    print("")
    # Get data function
    for _ in range(2):
        data = single_environment_manager.getData(batch_size=3, get_inputs=True, get_outputs=False)
        print("Data steps for {} : \n{}".format(single_environment_manager.environment.name.value, data['in']))
    print("")

    # Multiple environment management
    multiple_environment_config = SofaBaseEnvironmentConfig(environment_class=MyEnvironment,
                                                            simulations_per_step=1,
                                                            always_create_data=True,
                                                            multiprocessing=5,
                                                            multiprocess_method='process')
    multiple_environment_manager = EnvironmentManager(environment_config=multiple_environment_config)
    print(multiple_environment_config.getDescription())
    for env in multiple_environment_manager.environment:
        print(env.getDescription())

    for _ in range(2):
        data = multiple_environment_manager.getData(batch_size=1, get_inputs=True, get_outputs=False)
        for env in multiple_environment_manager.environment:
            print("Current step {} : {}".format(env.name, env.getInput()))

    return


if __name__ == '__main__':
    main()
