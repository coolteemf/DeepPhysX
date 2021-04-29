from DeepPhysX_Sofa.Environment.SofaBaseEnvironment import SofaBaseEnvironment
from DeepPhysX_Sofa.Environment.SofaBaseEnvironmentConfig import SofaBaseEnvironmentConfig


def main():

    # Single environment
    single_environment_config = SofaBaseEnvironmentConfig(environment_class=SofaBaseEnvironment,
                                                          simulations_per_step=1,
                                                          max_wrong_samples_per_step=10,
                                                          always_create_data=False,
                                                          multiprocessing=1)
    single_environment = single_environment_config.createEnvironment()
    print(single_environment_config.getDescription())
    print(single_environment.getDescription())

    # Multiple environment
    multiple_environment_config = SofaBaseEnvironmentConfig(environment_class=SofaBaseEnvironment,
                                                            simulations_per_step=5,
                                                            max_wrong_samples_per_step=6,
                                                            always_create_data=True,
                                                            multiprocessing=3)
    multiple_environment = multiple_environment_config.createEnvironment()
    print(multiple_environment_config.getDescription())
    for environment in multiple_environment:
        print(environment.getDescription())

    return


if __name__ == '__main__':
    main()