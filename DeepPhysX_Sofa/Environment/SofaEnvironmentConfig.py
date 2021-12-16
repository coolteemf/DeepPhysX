from os.path import join, dirname
from sys import modules
from subprocess import call as subprocesscall

import Sofa

from DeepPhysX_Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from .SofaEnvironment import SofaEnvironment


class SofaEnvironmentConfig(BaseEnvironmentConfig):

    def __init__(self,
                 environment_class=SofaEnvironment,
                 visualizer=None,
                 simulations_per_step=1,
                 screenshot_sample_rate=0,
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
        """
        SofaEnvironmentConfig is a configuration class to parameterize and create a SofaEnvironment for the
        EnvironmentManager.
        :param environment_class: Class from which an instance will be created
        :type environment_class: type[SofaEnvironment]
        :param visual_object: Class of the visual object which template visual data
        :param int simulations_per_step: Number of iterations to compute in the Environment at each time step
        :param int screenshot_sample_rate: A screenshot of the viewer will be done every x sample
        :param int max_wrong_samples_per_step: Maximum number of wrong samples to produce in a step
        :param bool always_create_data: If True, data will always be created from environment. If False, data will be
                                        created from the environment during the first epoch and then re-used from the
                                        Dataset.
        :param use_prediction_in_environment: If True, the prediction will always be used in the environment
        :param param_dict: Dictionary containing specific environment parameters
        :param as_tcpip_client: Environment is own by a TcpIpClient if True, by an EnvironmentManager if False
        :param int number_of_thread: Number of thread to run
        :param max_client_connection: Maximum number of handled instances
        :param environment_file: Path of the file containing the Environment class
        :param ip_address: IP address of the TcpIpObject
        :param port: Port number of the TcpIpObject
        """

        BaseEnvironmentConfig.__init__(self,
                                       environment_class=environment_class,
                                       visualizer=visualizer,
                                       simulations_per_step=simulations_per_step,
                                       screenshot_sample_rate=screenshot_sample_rate,
                                       max_wrong_samples_per_step=max_wrong_samples_per_step,
                                       always_create_data=always_create_data,
                                       use_prediction_in_environment=use_prediction_in_environment,
                                       as_tcpip_client=as_tcpip_client,
                                       number_of_thread=number_of_thread,
                                       max_client_connection=max_client_connection,
                                       environment_file=environment_file,
                                       param_dict=param_dict,
                                       ip_address=ip_address,
                                       port=port)

    def start_client(self, idx=1):
        """
        Run a subprocess to start a TcpIpClient.

        :param int idx: Index of client
        :return:
        """
        script = join(dirname(modules[SofaEnvironment.__module__].__file__), 'launcherSofaEnvironment.py')
        # Usage: python3 script.py <file_path> <environment_class> <ip_address> <port> <idx> <nb_threads>"
        subprocesscall(['python3', script,
                        self.environment_file,
                        self.environment_class.__name__,
                        self.ip_address,
                        str(self.port),
                        str(idx),
                        str(self.number_of_thread)])

    def createEnvironment(self, environment_manager):
        """
        Create an Environment that will not be a TcpIpObject.

        :param environment_manager: EnvironmentManager that handles the Environment
        :return: Environment
        """
        root_node = Sofa.Core.Node()
        environment = root_node.addObject(self.environment_class(environment_manager=environment_manager,
                                                                 root_node=root_node,
                                                                 as_tcpip_client=False))
        environment.create()
        environment.init()
        return environment
