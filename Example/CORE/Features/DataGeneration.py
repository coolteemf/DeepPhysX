"""
Dataset Generation
Run the pipeline DataGenerator to produce a Dataset only.
"""

# DeepPhysX related imports
from DeepPhysX_Core.Pipelines.BaseDataGenerator import BaseDataGenerator
from DeepPhysX_Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig

# Session related imports
from EnvironmentOffscreen import MeanEnvironmentOffscreen


def main():
    # Define the number of points and the dimension
    nb_points = 30
    dimension = 3
    # Environment configuration
    environment_config = BaseEnvironmentConfig(environment_class=MeanEnvironmentOffscreen,
                                               param_dict={'constant': True,
                                                           'nb_points': nb_points,
                                                           'dimension': dimension},
                                               as_tcp_ip_client=False)
    # Dataset configuration
    dataset_config = BaseDatasetConfig()
    # Create DataGenerator
    data_generator = BaseDataGenerator(session_name='sessions/data_generation',
                                       environment_config=environment_config,
                                       dataset_config=dataset_config,
                                       nb_batches=500,
                                       batch_size=10)
    # Launch the training session
    data_generator.execute()


if __name__ == '__main__':
    main()
