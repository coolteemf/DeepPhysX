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
from Environment.Armadillo import Armadillo
from Environment.parameters import p_model


def launch_runner():

    # Environment config
    env_config = BaseEnvironmentConfig(environment_class=Armadillo,
                                       visualizer=VedoVisualizer,
                                       as_tcp_ip_client=False,
                                       param_dict={'detailed': True,
                                                   'pattern': False})

    # UNet config
    nb_hidden_layers = 2
    nb_neurons = p_model.nb_nodes * 3
    layers_dim = [nb_neurons] + [nb_neurons for _ in range(nb_hidden_layers + 1)] + [nb_neurons]
    net_config = FCConfig(network_name='armadillo_FC',
                          dim_output=3,
                          dim_layers=layers_dim)

    # Dataset config
    dataset_config = BaseDatasetConfig(partition_size=1, shuffle_dataset=True)

    # Runner
    runner = BaseRunner(session_dir="sessions/armadillo",
                        dataset_config=dataset_config,
                        environment_config=env_config,
                        network_config=net_config,
                        nb_steps=0)
    runner.execute()
    runner.close()


if __name__ == '__main__':

    launch_runner()
