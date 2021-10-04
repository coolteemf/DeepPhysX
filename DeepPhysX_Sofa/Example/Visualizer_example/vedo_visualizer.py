
# Basic python imports
import sys

# DeepPhysX's Core imports
from DeepPhysX_Core.Manager.EnvironmentManager import EnvironmentManager
from DeepPhysX_Core.Visualizer.MeshVisualizer import MeshVisualizer

# DeepPhysX's Sofa imports
from DeepPhysX_Sofa.Environment.SofaEnvironmentConfig import SofaEnvironmentConfig, BytesNumpyConverter

from DeepPhysX_Sofa.Example.Visualizer_example import FEMBeam



def createScene(root_node=None):
    env_config = SofaEnvironmentConfig(environment_class=FEMBeam,
                                       environment_file=sys.modules[FEMBeam.__module__].__file__,
                                       number_of_thread=int(sys.argv[1]),
                                       socket_data_converter=BytesNumpyConverter,
                                       always_create_data=False)
    vedo_visualizer = MeshVisualizer()
    env_manager = EnvironmentManager(environment_config=env_config, visualizer=vedo_visualizer)

    return env_manager


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: python3 {sys.argv[0]} <nb_thread>")
        sys.exit(1)
    env = createScene()
    while True:
        env.step()
