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
from DeepPhysX_PyTorch.UNet.UNetConfig import UNetConfig

# Session related imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Environment.Armadillo import Armadillo
from Environment.parameters import grid_resolution


def launch_runner():

    # Environment config
    env_config = BaseEnvironmentConfig(environment_class=Armadillo,
                                       visualizer=VedoVisualizer,
                                       as_tcp_ip_client=False,
                                       param_dict={'detailed': False,
                                                   'pattern': True})

    # UNet config
    net_config = UNetConfig(network_name='armadillo_UNet',
                            input_size=grid_resolution,
                            nb_dims=3,
                            nb_input_channels=3,
                            nb_first_layer_channels=128,
                            nb_output_channels=3,
                            nb_steps=3,
                            two_sublayers=True,
                            border_mode='same',
                            skip_merge=False)

    # Dataset config
    dataset_config = BaseDatasetConfig(normalize=True)

    # Runner
    runner = BaseRunner(session_dir="sessions/armadillo_training_dpx",
                        dataset_config=dataset_config,
                        environment_config=env_config,
                        network_config=net_config,
                        nb_steps=0)
    runner.execute()
    runner.close()


if __name__ == '__main__':

    dpx_dataset, dpx_training = 'sessions/armadillo_data_dpx', 'sessions/armadillo_training_dpx'
    if not os.path.exists(dpx_dataset) or not os.path.exists(dpx_training):
        from download import download_all
        print('Downloading Demo training data to launch prediction...')
        download_all()

    launch_runner()