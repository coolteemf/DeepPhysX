"""
prediction.py
Launch the prediction session in a VedoVisualizer.
"""

# Python related imports
import os
import sys

# DeepPhysX related imports
from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX_Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX_Core.Visualizer.VedoVisualizer import VedoVisualizer
from DeepPhysX_Core.Pipelines.BaseRunner import BaseRunner
from DeepPhysX_PyTorch.FC.FCConfig import FCConfig

# Session related imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Environment.Beam import Beam
from Environment.parameters import p_model


def launch_runner():

    # Environment config
    env_config = BaseEnvironmentConfig(environment_class=Beam,
                                       visualizer=VedoVisualizer,
                                       as_tcp_ip_client=False,
                                       param_dict={'compute_sample': True})

    # UNet config
    nb_hidden_layers = 3
    nb_neurons = p_model.nb_nodes * 3
    layers_dim = [nb_neurons] + [nb_neurons for _ in range(nb_hidden_layers)] + [nb_neurons]
    net_config = FCConfig(network_name='beam_FC',
                          dim_output=3,
                          dim_layers=layers_dim,
                          biases=True)

    # Dataset config
    dataset_config = BaseDatasetConfig(normalize=True,
                                       dataset_dir='sessions/beam_data_dpx',
                                       use_mode='Validation')

    # Runner
    runner = BaseRunner(session_dir="sessions/beam_training_dpx",
                        dataset_config=dataset_config,
                        environment_config=env_config,
                        network_config=net_config,
                        nb_steps=0)
    runner.execute()
    runner.close()


if __name__ == '__main__':

    dpx_dataset, dpx_training = 'sessions/beam_data_dpx', 'sessions/beam_training_dpx'
    if not os.path.exists(dpx_dataset) or not os.path.exists(dpx_training):
        from download import download_all
        print('Downloading Demo training data to launch prediction...')
        download_all()

    launch_runner()
