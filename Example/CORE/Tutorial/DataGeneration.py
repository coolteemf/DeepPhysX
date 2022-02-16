"""
#04 - Data Generation
Only create a Dataset without training session.
"""

# DeepPhysX related imports
from DeepPhysX_Core.Pipelines.BaseDataGenerator import BaseDataGenerator

# Tutorial related imports
from Configuration import env_config, dataset_config


def main():
    # Create the Pipeline
    pipeline = BaseDataGenerator(session_name='sessions/tutorial_data_generation',
                                 dataset_config=dataset_config,
                                 environment_config=env_config,
                                 nb_batches=100,
                                 batch_size=10)
    # Launch the Pipeline
    pipeline.execute()


if __name__ == '__main__':
    main()