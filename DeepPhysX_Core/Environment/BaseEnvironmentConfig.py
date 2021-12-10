from os import cpu_count
from os.path import join, dirname
from threading import Thread
from subprocess import run as subprocessRun
from sys import modules

from DeepPhysX_Core.AsyncSocket.TcpIpServer import TcpIpServer
from DeepPhysX_Core.Environment.BaseEnvironment import BaseEnvironment


class BaseEnvironmentConfig:

    def __init__(self,
                 environment_class=None,
                 visualizer=None,
                 simulations_per_step=1,
                 max_wrong_samples_per_step=10,
                 always_create_data=False,
                 record_wrong_samples=False,
                 use_prediction_in_environment=False,
                 param_dict={},
                 as_tcpip_client=True,
                 number_of_thread=1,
                 max_client_connection=1000,
                 environment_file=None,
                 ip_address='localhost',
                 port=10000):
<<<<<<< HEAD
=======

>>>>>>> ByteConverter is now a default member
        """
        BaseEnvironmentConfig is a configuration class to parameterize and create a BaseEnvironment for the
        EnvironmentManager.

        :param environment_class: Class from which an instance will be created
        :type environment_class: type[BaseEnvironment]
        :param visual_object: Class of the visual object which template visual data
        :param int simulations_per_step: Number of iterations to compute in the Environment at each time step
        :param int max_wrong_samples_per_step: Maximum number of wrong samples to produce in a step
        :param bool always_create_data: If True, data will always be created from environment. If False, data will be
                                        created from the environment during the first epoch and then re-used from the
                                        Dataset.
        :param bool record_wrong_samples: If True, wrong samples are recorded through Visualizer
        :param use_prediction_in_environment: If True, the prediction will always be used in the environment
        :param param_dict: Dictionary containing specific environment parameters
        :param as_tcpip_client: Environment is own by a TcpIpClient if True, by an EnvironmentManager if False
        :param int number_of_thread: Number of thread to run
        :param max_client_connection: Maximum number of handled instances
        :param environment_file: Path of the file containing the Environment class
        :param ip_address: IP address of the TcpIpObject
        :param port: Port number of the TcpIpObject
        """

        self.name = self.__class__.__name__

        # Check simulations_per_step type and value
        if type(simulations_per_step) != int:
            raise TypeError(f"[{self.name}] Wrong simulations_per_step type: int required, get "
                            f"{type(simulations_per_step)}")
        if simulations_per_step < 1:
            raise ValueError(f"[{self.name}] Given simulations_per_step value is negative or null")
        # Check max_wrong_samples_per_step type and value
        if type(max_wrong_samples_per_step) != int:
            raise TypeError(f"[{self.name}] Wrong max_wrong_samples_per_step type: int required, get "
                            f"{type(max_wrong_samples_per_step)}")
        if simulations_per_step < 1:
            raise ValueError(f"[{self.name}] Given max_wrong_simulations_per_step value is negative or null")
        # Check always_create_data type
        if type(always_create_data) != bool:
            raise TypeError(f"[{self.name}] Wrong always_create_data type: bool required, get "
                            f"{type(always_create_data)}")

        if type(number_of_thread) != int and number_of_thread < 0:
            raise TypeError(f"[{self.name}] The number_of_thread number must be a positive integer.")
        self.max_client_connections = max_client_connection


        # TcpIpClients parameterization
        self.environment_class = environment_class
        self.environment_file = environment_file if environment_file is not None else modules[self.environment_class.__module__].__file__
        self.param_dict = param_dict
        self.as_tcpip_client = as_tcpip_client

        # EnvironmentManager parameterization
        self.received_parameters = {}
        self.always_create_data = always_create_data
        self.record_wrong_samples = record_wrong_samples
        self.use_prediction_in_environment = use_prediction_in_environment
        self.simulations_per_step = simulations_per_step
        self.max_wrong_samples_per_step = max_wrong_samples_per_step
        self.visualizer = visualizer

        # TcpIpServer parameterization
        self.ip_address = ip_address
        self.port = port
        self.server_is_ready = False
        self.number_of_thread = min(max(number_of_thread, 1), cpu_count())  # Assert nb is between 1 and cpu_count

    def createServer(self, environment_manager=None, batch_size=1):
        """
        Create a TcpIpServer and launch TcpIpClients in subprocesses.

        :param environment_manager: EnvironmentManager
        :param int batch_size: Number of sample in a batch
        :return: TcpIpServer
        """
        # Create server
<<<<<<< HEAD
        server = TcpIpServer(max_client_count=self.max_client_connections, batch_size=batch_size,
                             nb_client=self.number_of_thread, manager=environment_manager,
                             ip_address=self.ip_address, port=self.port)
=======
        server = TcpIpServer(max_client_count=self.max_client_connections,
                             batch_size=batch_size, nb_client=self.number_of_thread, manager=environment_manager)
>>>>>>> ByteConverter is now a default member

        server_thread = Thread(target=self.start_server, args=(server,))
        server_thread.start()

        # Create clients
        client_threads = []
        for i in range(self.number_of_thread):
            client_thread = Thread(target=self.start_client, args=(i,))
            client_threads.append(client_thread)
        for client in client_threads:
            client.start()

        # Return server to manager when ready
        while not self.server_is_ready:
            pass
        return server

    def start_server(self, server):
        """
        Start TcpIpServer.

        :param server: TcpIpServer
        :return:
        """
        # Allow clients connections
        server.connect()
        # Send and receive parameters with clients
        self.received_parameters = server.initialize(self.param_dict)
        # Server is ready
        self.server_is_ready = True

    def start_client(self, idx=1):
        """
        Run a subprocess to start a TcpIpClient.

        :param int idx: Index of client
        :return:
        """
        script = join(dirname(modules[BaseEnvironment.__module__].__file__),
                              'launcherBaseEnvironment.py')
        # Usage: python3 script.py <file_path> <environment_class> <ip_address> <port> <idx> <nb_threads>"
        subprocessRun(['python3',
                        script,
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
        # Create instance
        environment = self.environment_class(environment_manager=environment_manager, as_tcpip_client=False,
<<<<<<< HEAD
                                             visual_object=self.visual_object)
        # Create environment objects and initialize
=======
                                             visual_object=self.visualizer)
>>>>>>> Environment update
        environment.create()
        environment.init()
        return environment

    def __str__(self):
        """
        :return: String containing information about the BaseEnvironmentConfig object
        """
        # Todo: fields in Configs are the set in Managers or objects, remove __str__ method
        description = "\n"
        description += f"{self.name}\n"
        description += f"    Environment class: {self.environment_class.__name__}\n"
        description += f"    Simulations per step: {self.simulations_per_step}\n"
        description += f"    Max wrong samples per step: {self.max_wrong_samples_per_step}\n"
        description += f"    Always create data: {self.always_create_data}\n"
        return description
