"""
training.py
Launch the training session with a VedoVisualizer.
Use 'python3 training.py' to run the pipeline with existing samples from a Dataset (default).
Use 'python3 training.py <nb_thread>' to run the pipeline with newly created samples in Environment.
"""

# Python related imports
import os.path
import sys
import torch

# DeepPhysX related imports
from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX_Core.Visualizer.VedoVisualizer import VedoVisualizer
from DeepPhysX_Core.Pipelines.BaseTrainer import BaseTrainer
from DeepPhysX_PyTorch.FC.FCConfig import FCConfig
from DeepPhysX_Sofa.Environment.SofaEnvironmentConfig import SofaEnvironmentConfig

# Working session imports
from Environment.LiverTraining import LiverTraining, np

# Training parameters
nb_epochs = 400
nb_batch = 500
batch_size = 32
lr = 1e-5


def launch_trainer(dataset_dir, nb_env):

    # Environment config
    env_config = SofaEnvironmentConfig(environment_class=LiverTraining,
                                       visualizer=VedoVisualizer,
                                       number_of_thread=nb_env)

    # Get the data size
    env_instance = env_config.create_environment(None)
    input_size, output_size = env_instance.input_size, env_instance.output_size
    env_instance.close()
    del env_instance

    # FC config
    nb_hidden_layers = 3
    nb_neurons = np.multiply(*input_size)
    nb_final_neurons = np.multiply(*output_size)
    layers_dim = [nb_neurons] + [nb_neurons for _ in range(nb_hidden_layers)] + [nb_final_neurons]
    net_config = FCConfig(network_name='liver_FC',
                          loss=torch.nn.MSELoss,
                          lr=lr,
                          optimizer=torch.optim.Adam,
                          dim_output=3,
                          dim_layers=layers_dim,
                          biases=True)

    # Dataset config
    dataset_config = BaseDatasetConfig(partition_size=1,
                                       shuffle_dataset=True,
                                       normalize=True,
                                       dataset_dir=dataset_dir)

    # Trainer
    trainer = BaseTrainer(session_name="sessions/liver_training_user",
                          dataset_config=dataset_config,
                          environment_config=env_config,
                          network_config=net_config,
                          nb_epochs=nb_epochs,
                          nb_batches=nb_batch,
                          batch_size=batch_size)

    # Launch the training session
    trainer.execute()


if __name__ == '__main__':

    # Check data
    if not os.path.exists('Environment/models'):
        from download import download_all
        print('Downloading Liver demo data...')
        download_all()

    # Define dataset
    dpx_session = 'sessions/liver_data_dpx'
    user_session = 'sessions/liver_data_user'
    # Take user dataset by default
    dataset = user_session if os.path.exists(user_session) else dpx_session

    # Get nb_thread options
    nb_thread = 1
    if len(sys.argv) > 1:
        try:
            nb_thread = int(sys.argv[1])
        except ValueError:
            print("Script option must be an integer <nb_sample> for samples produced in Environment(s)."
                  "By default, samples are loaded from an existing Dataset.")
            quit(0)

    # Launch pipeline
    launch_trainer(dataset, nb_thread)
