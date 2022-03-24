"""
training.py
Launch the training session with a VedoVisualizer.
Use 'python3 training.py <nb_thread>' to run the pipeline with newly created samples in Environment (default).
Use 'python3 training.py -d' to run the pipeline with existing samples from a Dataset.
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
from Environment.ArmadilloTraining import ArmadilloTraining
import Environment.parameters as parameters

# Training parameters
nb_epochs = 200
nb_batch = 800
batch_size = 20
lr = 1e-5


def launch_trainer(dataset_dir, nb_env):

    # Environment config
    env_config = SofaEnvironmentConfig(environment_class=ArmadilloTraining,
                                       visualizer=VedoVisualizer,
                                       number_of_thread=nb_env)

    # UNet config
    nb_hidden_layers = 2
    nb_neurons = parameters.p_model.nb_nodes * 3
    layers_dim = [nb_neurons] + [nb_neurons for _ in range(nb_hidden_layers + 1)] + [nb_neurons]
    net_config = FCConfig(network_name='armadillo_FC',
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
    trainer = BaseTrainer(session_name="sessions/armadillo_training_user",
                          dataset_config=dataset_config,
                          environment_config=env_config,
                          network_config=net_config,
                          nb_epochs=nb_epochs,
                          nb_batches=nb_batch,
                          batch_size=batch_size)

    # Launch the training session
    trainer.execute()


if __name__ == '__main__':

    # Get dataset and nb_thread options
    dataset, nb_thread = None, 1
    if len(sys.argv) > 1:

        # Run with Dataset
        if sys.argv[1] == '-d':
            # Check dataset existence
            dpx_session, user_session = 'sessions/armadillo_data_dpx', 'sessions/armadillo_data_user'
            if not os.path.exists(dpx_session) and not os.path.exists(user_session):
                from download import download_dataset
                print('Downloading Demo dataset to launch training...')
                download_dataset()
            dataset = dpx_session if os.path.exists(dpx_session) else user_session

        # Run with Environment(s)
        else:
            try:
                nb_thread = int(sys.argv[1])
            except ValueError:
                print("Script option must be either an integer <nb_sample> for samples produced in Environment(s) "
                      "(default) or '-d' for samples loaded from an existing Dataset.")
                quit(0)

    # Launch pipeline
    launch_trainer(dataset, nb_thread)
