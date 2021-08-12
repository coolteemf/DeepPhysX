import sys
import time

from DeepPhysX_Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX_Core.AsyncSocket.BytesNumpyConverter import BytesNumpyConverter
from DeepPhysX_Core.Manager.EnvironmentManager import EnvironmentManager
from Tests.AsyncSocket.Environment import Environment

if len(sys.argv) != 2:
    print(f"Usage: python3 {sys.argv[0]} <nb_thread>")
    sys.exit(1)


env_config = BaseEnvironmentConfig(environment_class=Environment, environment_file=sys.modules[Environment.__module__].__file__,
                                   number_of_thread=int(sys.argv[1]),
                                   socket_data_converter=BytesNumpyConverter)
env_manager = EnvironmentManager(environment_config=env_config, batch_size=2)
for i in range(3):
    data = env_manager.getData(get_inputs=True, get_outputs=True, animate=True)
    print()
    print(f"Batch n°{i}", data)
    print()
env_manager.close()
