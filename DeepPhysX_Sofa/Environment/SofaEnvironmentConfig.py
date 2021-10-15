from os.path import join, dirname
from sys import modules
from subprocess import call as subprocesscall

import Sofa

from DeepPhysX_Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from .SofaEnvironment import SofaEnvironment


class SofaEnvironmentConfig(BaseEnvironmentConfig):

    def __init__(self,
                 environment_class=SofaEnvironment,
                 visual_object=None,
                 simulations_per_step=1,
                 max_wrong_samples_per_step=10,
                 always_create_data=False,
                 use_prediction_in_environment=False,
                 number_of_thread=1,
                 as_tcpip_client=True,
                 max_client_connection=1000,
                 environment_file='',
                 param_dict={},
                 ip_address='localhost',
                 port=10000):

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
                                       param_dict=param_dict, ip_address=ip_address, port=port)

    def start_client(self, idx=1):
        script = join(dirname(modules[SofaEnvironment.__module__].__file__),
                              'launcherSofaEnvironment.py')
        # Usage: python3 script.py <file_path> <environment_class> <ip_address> <port> <converter_class> <idx>"
        subprocesscall(['python3', script,
                        self.environment_file,
                        self.environment_class.__name__,
                        self.ip_address,
                        str(self.port),
                        str(idx),
                        str(self.number_of_thread)])


    def createEnvironment(self, environment_manager):
        """

        :return:
        """
        root_node = Sofa.Core.Node()
        environment = root_node.addObject(self.environment_class(environment_manager=environment_manager,
                                                                 root_node=root_node, as_tcpip_client=False,
                                                                 visual_object=self.visual_object))
        environment.create()
        environment.init()
        return environment
