"""
dataset.py
Run the pipeline DataGenerator to produce a Dataset only.
Use 'python3 dataset.py -t' to produce training Dataset (default).
Use 'python3 dataset.py -v' to produce validation Dataset.
"""

# Python related imports
import sys

# DeepPhysX related imports
from DeepPhysX_Core.Pipelines.BaseDataGenerator import BaseDataGenerator
from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX_Core.Visualizer.VedoVisualizer import VedoVisualizer
from DeepPhysX_Sofa.Environment.SofaEnvironmentConfig import SofaEnvironmentConfig

# Session related imports
from Environment.LiverTraining import LiverTraining

# Dataset parameters
nb_batches = {'Training': 500, 'Validation': 50}
batch_size = {'Training': 32, 'Validation': 10}


def launch_data_generation(dataset_mode):

    # Environment configuration
    environment_config = SofaEnvironmentConfig(environment_class=LiverTraining,
                                               visualizer=VedoVisualizer,
                                               as_tcp_ip_client=True,
                                               number_of_thread=10)

    # Dataset configuration
    dataset_config = BaseDatasetConfig(partition_size=1, shuffle_dataset=True, use_mode=dataset_mode)

    # Create DataGenerator
    data_generator = BaseDataGenerator(session_name='sessions/liver_data_user',
                                       environment_config=environment_config,
                                       dataset_config=dataset_config,
                                       nb_batches=nb_batches[dataset_mode],
                                       batch_size=batch_size[dataset_mode])

    # Launch the data generation session
    data_generator.execute()


if __name__ == '__main__':

    # Check data
    if not os.path.exists('Environment/models'):
        from download import download_all
        print('Downloading Demo data...')
        download_all()

    # Get dataset mode
    mode = 'Training'
    if len(sys.argv) > 1:
        if sys.argv[1] not in ['-t', '-v']:
            print("Script option must be either '-t' for training dataset (default) or '-v' for validation dataset.")
            quit(0)
        mode = 'Validation' if sys.argv[1] == '-v' else 'Training'

    # Launch pipeline
    launch_data_generation(mode)
