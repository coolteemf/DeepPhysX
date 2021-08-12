import os
import sys
import subprocess

from DeepPhysX_Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig, BytesNumpyConverter
from .SofaEnvironment import SofaEnvironment


class SofaEnvironmentConfig(BaseEnvironmentConfig):

    def __init__(self, environment_class=SofaEnvironment, simulations_per_step=1, max_wrong_samples_per_step=10,
                 always_create_data=False, number_of_thread=1, max_client_connection=1000, environment_file='',
                 param_dict={}, ip_address='localhost', port=10000, socket_data_converter=BytesNumpyConverter):

        BaseEnvironmentConfig.__init__(self, environment_class=environment_class,
                                       simulations_per_step=simulations_per_step,
                                       max_wrong_samples_per_step=max_wrong_samples_per_step,
                                       always_create_data=always_create_data,
                                       number_of_thread=number_of_thread,
                                       max_client_connection=max_client_connection,
                                       environment_file=environment_file,
                                       param_dict=param_dict, ip_address=ip_address, port=port,
                                       socket_data_converter=socket_data_converter)

    def start_client(self, idx=1):
        script = os.path.join(os.path.dirname(sys.modules[SofaEnvironment.__module__].__file__),
                              'launcherSofaEnvironment.py')
        # Usage: python3 script.py <file_path> <environment_class> <ip_address> <port> <converter_class> <idx>"
        subprocess.run(['python3', script, self.environment_file, self.environment_class.__name__,
                        self.ip_address, str(self.port), self.socket_data_converter.__name__, str(idx)])