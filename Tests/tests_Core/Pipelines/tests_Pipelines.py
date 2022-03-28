import os
import unittest
import tempfile

from DeepPhysX_Core.Pipelines.BaseYamlLoader import BaseYamlLoader

from DeepPhysX_Core.Pipelines.BaseYamlExporter import BaseYamlExporter

from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX_Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX_Core.Network.BaseNetworkConfig import BaseNetworkConfig
from Example.CORE.Tutorial.T1_environment import DummyEnvironment
from Example.CORE.Tutorial.T2_network import DummyNetwork, DummyOptimization


class TestPipelines(unittest.TestCase):

    def setUp(self):
        # Create the Environment config
        self.env_config = (BaseEnvironmentConfig, dict(environment_class=DummyEnvironment,  # The Environment class to create
                                                  visualizer=None,  # The Visualizer to use
                                                  simulations_per_step=1,  # The number of sub-steps to run
                                                  use_dataset_in_environment=False,
                                                  # Dataset will not be sent to Environment
                                                  param_dict={'increment': 1},  # Parameters to send at init
                                                  as_tcp_ip_client=True,  # Create a Client / Server architecture
                                                  number_of_thread=3,  # Number of Clients connected to Server
                                                  ip_address='localhost',  # IP address to use for communication
                                                  port=10001))
        self.env_config_obj = self.env_config[0](**self.env_config[1])
        # Create the Network config
        self.net_config = (BaseNetworkConfig, dict(network_class=DummyNetwork,  # The Network class to create
                                              optimization_class=DummyOptimization,  # The Optimization class to create
                                              network_name='DummyNetwork',  # Nickname of the Network
                                              network_type='Dummy',  # Type of the Network
                                              save_each_epoch=False,  # Do not save the network at each epoch
                                              require_training_stuff=False,  # loss and optimizer can remain at None
                                              lr=None,  # Learning rate
                                              loss=None,  # Loss class
                                              optimizer=None))  # Optimizer class
        self.net_config_obj = self.net_config[0](**self.net_config[1])
        # Create the Dataset config
        self.dataset_config = (BaseDatasetConfig, dict(partition_size=1,  # Max size of the Dataset
                                                  shuffle_dataset=False))  # Dataset should be shuffled
        self.dataset_config_obj = self.dataset_config[0](**self.dataset_config[1])



    def test_export_load_roundtrip(self):
        pipeline_config = dict(
            session_dir=os.getcwd(),
            session_name = 'sessions/tutorial_offline_training',
            environment_config=self.env_config,
            dataset_config=self.dataset_config,
            network_config=self.net_config,
            nb_epochs=2,
            nb_batches=100,
            batch_size=10,
        )
        tf = tempfile.NamedTemporaryFile(suffix=".yml")
        BaseYamlExporter(tf.name, pipeline_config)
        loaded_pipeline_config = BaseYamlLoader(tf.name)
        self.assertEqual(pipeline_config, loaded_pipeline_config)
