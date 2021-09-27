import os
import sys
import subprocess

import Sofa

from DeepPhysX_Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig, BytesNumpyConverter
from .SofaEnvironment import SofaEnvironment
from DeepPhysX_Core.Visualizer.VedoObject import VedoObject


class SofaEnvironmentConfig(BaseEnvironmentConfig):

    def __init__(self, environment_class=SofaEnvironment, visual_object=VedoObject,
                 simulations_per_step=1, max_wrong_samples_per_step=10,
                 always_create_data=False, use_prediction_in_environment=False, as_tcpip_client=True,
                 number_of_thread=1, max_client_connection=1000, environment_file='',
                 param_dict={}, ip_address='localhost', port=10000, socket_data_converter=BytesNumpyConverter):

        BaseEnvironmentConfig.__init__(self, environment_class=environment_class,
                                       visual_object=visual_object,
                                       simulations_per_step=simulations_per_step,
                                       max_wrong_samples_per_step=max_wrong_samples_per_step,
                                       always_create_data=always_create_data,
                                       use_prediction_in_environment=use_prediction_in_environment,
                                       as_tcpip_client=as_tcpip_client,
                                       number_of_thread=number_of_thread,
                                       max_client_connection=max_client_connection,
                                       environment_file=environment_file,
                                       param_dict=param_dict, ip_address=ip_address, port=port,
                                       socket_data_converter=socket_data_converter)

    def start_client(self, idx=1):
        script = os.path.join(os.path.dirname(sys.modules[SofaEnvironment.__module__].__file__),
                              'launcherSofaEnvironment.py')
        # Usage: python3 script.py <file_path> <environment_class> <ip_address> <port> <converter_class> <idx>"
        subprocess.run(['python3', script,
                        self.environment_file,
                        self.environment_class.__name__,
                        self.ip_address,
                        str(self.port),
                        self.socket_data_converter.__name__,
                        str(idx)])

    def createEnvironment(self, environment_manager):
        """

        :return:
        """
        root_node = Sofa.Core.Node()
        environment = root_node.addObject(self.environment_class(environment_manager=environment_manager, root_node=root_node, as_tcpip_client=False,
                                                                 visual_object=self.visual_object))
        environment.create()
        environment.init()
        return environment
