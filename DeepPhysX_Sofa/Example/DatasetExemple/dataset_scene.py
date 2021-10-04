# Basic python imports
import sys

# DeepPhysX's Core imports
from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX_Core.Visualizer.MeshVisualizer import MeshVisualizer
from DeepPhysX_Core.Pipelines.BaseDataGenerator import BaseDataGenerator
from DeepPhysX_Sofa.Example.Training_example.EnvironmentSofa import FEMBeam

# DeepPhysX's Sofa imports
from DeepPhysX_Sofa.Environment.SofaEnvironmentConfig import SofaEnvironmentConfig, BytesNumpyConverter


def createScene():
    env_config = SofaEnvironmentConfig(environment_class=FEMBeam,
                                       environment_file=sys.modules[FEMBeam.__module__].__file__,
                                       number_of_thread=int(sys.argv[1]),
                                       socket_data_converter=BytesNumpyConverter,
                                       always_create_data=False)

    dataset_config = BaseDatasetConfig(partition_size=1, shuffle_dataset=True)

    data_generator = BaseDataGenerator(session_name='training/example',
                                       dataset_config=dataset_config,
                                       environment_config=env_config,
                                       visualizer_class=MeshVisualizer,
                                       nb_batches=10, batch_size=1)
    data_generator.execute()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: python3 {sys.argv[0]} <nb_thread>")
        sys.exit(1)
    createScene()
