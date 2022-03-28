"""
#05 - Offline Training
Launch a training session with an existing Dataset.
"""

# Python related imports
import copy
import os

# DeepPhysX related imports
from DeepPhysX_Core.Pipelines.BaseYamlExporter import BaseYamlExporter
from DeepPhysX_Core.Pipelines.BaseTrainer import BaseTrainer
from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig

# Tutorial related imports
from T3_configuration import env_config, net_config


def launch_training():
    # Adapt the Dataset config with the existing dataset directory
    dataset_config = BaseDatasetConfig(dataset_dir=os.path.join(os.getcwd(), 'sessions/tutorial_data_generation'),
                                       partition_size=1,
                                       shuffle_dataset=False)
    # Create the Pipeline
    pipeline_config = dict(
        session_dir=os.getcwd(),
        session_name = 'sessions/tutorial_offline_training',
        environment_config=env_config,
        dataset_config=dataset_config,
        network_config=net_config,
        nb_epochs=2,
        nb_batches=100,
        batch_size=10,
    )
    pipeline = BaseTrainer(**pipeline_config)
    #export the parameters to a conf file
    conf_export_dir = os.path.join(pipeline.manager.session_dir,'conf.yml')
    pipeline_config_exported = BaseYamlExporter(conf_export_dir, pipeline_config)
    # Launch the Pipeline
    pipeline.execute()


if __name__ == '__main__':
    launch_training()
