import os
import time
import threading
import subprocess
import sys

from DeepPhysX_Core.Environment.BaseEnvironment import BaseEnvironment
from DeepPhysX_Core.AsyncSocket.TcpIpServer import TcpIpServer, BytesNumpyConverter
from DeepPhysX_Core.Visualizer.VedoVisualizer import VedoVisualizer


class BaseEnvironmentConfig:

    def __init__(self, environment_class=BaseEnvironment, simulations_per_step=1, max_wrong_samples_per_step=10,
                 always_create_data=False, number_of_thread=1, max_client_connection=1000, environment_file='',
                 param_dict={}, ip_address='localhost', port=10000, socket_data_converter=BytesNumpyConverter):
        """
        BaseEnvironmentConfig is a configuration class to parameterize and create a BaseEnvironment for the
        EnvironmentManager.

        :param environment_class: Class from which an instance will be created
        :type environment_class: type[BaseEnvironment]
        :param int simulations_per_step: Number of iterations to compute in the Environment at each time step
        :param int max_wrong_samples_per_step: Maximum number of wrong samples to produce in a step
        :param bool always_create_data: If True, data will always be created from environment. If False, data will be
                                        created from the environment during the first epoch and then re-used from the
                                        Dataset.
        :param int number_of_thread: Number of thread to run
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
            raise TypeError(f"[{self.name}] The number_of_thread number must be a positive int.")
        self.socket_data_converter = socket_data_converter
        self.max_client_connections = max_client_connection
        self.environment_file = environment_file

        # TcpIpClients parameterization
        self.environment_class = environment_class
        self.param_dict = param_dict

        # EnvironmentManager parameterization
        self.received_parameters = {}
        self.always_create_data = always_create_data
        self.simulations_per_step = simulations_per_step
        self.max_wrong_samples_per_step = max_wrong_samples_per_step

        # TcpIpServer parameterization
        self.ip_address = ip_address
        self.port = port
        self.server_is_ready = False
        self.number_of_thread = min(max(number_of_thread, 1), os.cpu_count())  # Assert nb is between 1 and cpu_count

    def createServer(self, environment_manager=None, batch_size=1):
        """
        Create a TcpIpServer and launch TcpIpClients in subprocesses.

        :param environment_manager: EnvironmentManager
        :param int batch_size: Number of sample in a batch
        :return: TcpIpServer
        """
        # Create server
        server = TcpIpServer(data_converter=self.socket_data_converter, max_client_count=self.max_client_connections,
                             batch_size=batch_size, nb_client=self.number_of_thread)
        server.manager = environment_manager

        server_thread = threading.Thread(target=self.start_server, args=(server,))
        server_thread.start()

        # Create clients
        client_threads = []
        for i in range(self.number_of_thread):
            client_thread = threading.Thread(target=self.start_client, args=(i,))
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
        script = os.path.join(os.path.dirname(sys.modules[BaseEnvironment.__module__].__file__),
                              'launcherBaseEnvironment.py')
        # Usage: python3 script.py <file_path> <environment_class> <ip_address> <port> <converter_class> <idx>"
        subprocess.run(['python3', script, self.environment_file, self.environment_class.__name__,
                        self.ip_address, str(self.port), self.socket_data_converter.__name__, str(idx)])

    def __str__(self):
        """
        :return: String containing information about the BaseEnvironmentConfig object
        """
        description = "\n"
        description += f"{self.name}\n"
        description += f"    Environment class: {self.environment_class.__name__}\n"
        description += f"    Simulations per step: {self.simulations_per_step}\n"
        description += f"    Max wrong samples per step: {self.max_wrong_samples_per_step}\n"
        description += f"    Always create data: {self.always_create_data}\n"
        return description
