import sys

from DeepPhysX_Sofa.Environment.SofaEnvironmentConfig import SofaEnvironmentConfig, BytesNumpyConverter
from DeepPhysX_Core.Manager.EnvironmentManager import EnvironmentManager
from Tests.AsyncSocket.EnvironmentSofa import EnvironmentSofa

if len(sys.argv) != 2:
    print(f"Usage: python3 {sys.argv[0]} <nb_thread>")
    sys.exit(1)

env_config = SofaEnvironmentConfig(environment_class=EnvironmentSofa,
                                   environment_file=sys.modules[EnvironmentSofa.__module__].__file__,
                                   number_of_thread=int(sys.argv[1]),
                                   socket_data_converter=BytesNumpyConverter)
env_manager = EnvironmentManager(environment_config=env_config, batch_size=2)
for i in range(3):
    data = env_manager.getData(get_inputs=True, get_outputs=True, animate=True)
    print()
    print(f"Batch n°{i}", data)
    print()
env_manager.close()
