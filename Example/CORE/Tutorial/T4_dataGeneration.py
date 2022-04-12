"""
#04 - Data Generation
Only create a Dataset without training session.
"""

# DeepPhysX related imports
import os

from DeepPhysX_Core.Pipelines.BaseDataGenerator import BaseDataGenerator

# Tutorial related imports
from T3_configuration import env_config, dataset_config


def launch_data_generation():
    # Create the Pipeline
    pipeline = BaseDataGenerator(session_dir=os.getcwd(),
                                 session_name='sessions/tutorial_data_generation',
                                 dataset_config=dataset_config[0](**dataset_config[1]),
                                 environment_config=env_config[0](**env_config[1]),
                                 nb_batches=100,
                                 batch_size=10)
    # Launch the Pipeline
    pipeline.execute()


if __name__ == '__main__':
    launch_data_generation()
