from asyncio import get_event_loop, run, gather
import numpy as np
from queue import SimpleQueue

from DeepPhysX_Core.AsyncSocket.TcpIpObject import TcpIpObject


class TcpIpServer(TcpIpObject):

    def __init__(self,
                 ip_address='localhost',
                 port=10000,
                 nb_client=5,
                 max_client_count=10,
                 batch_size=5,
                 manager=None):
        """
        TcpIpServer is used to communicate with clients associated with Environment to produce batches for the
        EnvironmentManager.

        :param str ip_address: IP address of the TcpIpObject
        :param int port: Port number of the TcpIpObject
        :param int nb_client: Number of expected client connections
        :param int max_client_count: Maximum number of allowed clients
        :param int batch_size: Number of samples in a batch
        :param manager: EnvironmentManager that handles the TcpIpServer
        """

        super(TcpIpServer, self).__init__(ip_address=ip_address,
                                          port=port)

        # Bind to server address
        print(f"[{self.name}] Binding to IP Address: {ip_address} on PORT: {port} with maximum client count: "
              f"{max_client_count}")
        self.sock.bind((ip_address, port))
        self.sock.listen(max_client_count)
        self.sock.setblocking(False)

        # Expect a defined number of clients
        self.clients = []
        self.nb_client = min(nb_client, max_client_count)

        # Init data to communicate with EnvironmentManager and Clients
        self.batch_size = batch_size
        self.data_fifo = SimpleQueue()
        self.data_dict = {}
        self.sample_to_client_id = []
        self.batch_from_dataset = None
        self.first_time = True

        # Reference to EnvironmentManager
        self.environment_manager = manager

    def connect(self):
        """
        Run __connect method with asyncio.

        :return:
        """

        print(f"[{self.name}] Waiting for clients...")
        run(self.__connect())

    async def __connect(self):
        """
        Accept connections from clients.

        :return:
        """

        loop = get_event_loop()
        # Accept clients connections one by one
        for _ in range(self.nb_client):
            client, _ = await loop.sock_accept(self.sock)
            label, client_id = await self.receive_labeled_data(loop=loop, sender=client)
            print(f"[{self.name}] Client n°{client_id} connected: {client}")
            self.clients.append([client_id, client])

    def initialize(self, param_dict):
        """
        Run __initialize method with asyncio.

        :param dict param_dict: Dictionary of parameters to send to the client's environment
        :return:
        """

        print(f"[{self.name}] Initializing clients...")
        run(self.__initialize(param_dict))

    async def __initialize(self, param_dict):
        """
        Send parameters to the clients to create their environments, receive parameters from clients in exchange.

        :param dict param_dict: Dictionary of parameters to send to the client's environment
        :return: Dictionary of parameters for each environment to send the manager
        """

        loop = get_event_loop()
        # Empty dictionaries for received parameters from clients
        self.data_dict = {client_ID: {} for client_ID in range(len(self.clients))}
        # Initialisation process for each client
        for client_id, client in self.clients:
            # Send parameters to client
            await self.send_dict(name="parameters", dict_to_send=param_dict, loop=loop, receiver=client)
            # Receive visualization data and parameters
            await self.listen_while_not_done(loop=loop, sender=client, data_dict=self.data_dict, client_id=client_id)
            print(f"[{self.name}] Client n°{client_id} initialisation done")

    def getBatch(self, get_inputs=True, get_outputs=True, animate=True):
        """
        Build a batch from clients samples.

        :param bool get_inputs: If True, compute and return input
        :param bool get_outputs: If True, compute and return output
        :param bool animate: If True, triggers an environment step
        :return: Batch (list of samples) & additional data in a dictionary
        """

        # Trigger communication protocol
        run(self.__request_data_from_client(get_inputs=get_inputs, get_outputs=get_outputs, animate=animate))

        # Sort stored data between following fields
        data_sorter = {'input': [], 'dataset_in': {}, 'output': [], 'dataset_out': {}, 'loss': []}
        # Map produced samples with clients ID
        self.sample_to_client_id = []

        # Process while queue is empty or batch is full
        while max([len(data_sorter[k]) for k in data_sorter]) < self.batch_size and not self.data_fifo.empty():
            # Get data dict from queue
            data = self.data_fifo.get()
            # Network in / out / loss
            for field in ['input', 'output', 'loss']:
                if field in data:
                    data_sorter[field].append(data[field])
            # Additional in / out
            for field in ['dataset_in', 'dataset_out']:
                if field in data:
                    for key in data[field]:
                        if key not in data_sorter[field].keys():
                            data_sorter[field][key] = []
                        data_sorter[field][key].append(data[field][key])
            # ID of client
            if 'ID' in data:
                self.sample_to_client_id.append(data['ID'])

        # Reformat data dict
        self.data_dict['loss'] = data_sorter['loss']
        self.data_dict['dataset_in'] = data_sorter['dataset_in']
        self.data_dict['dataset_out'] = data_sorter['dataset_out']
        return [data_sorter['input'], data_sorter['output']], self.data_dict

    async def __request_data_from_client(self, get_inputs=True, get_outputs=True, animate=True):
        """
        Trigger a communication protocol for each client. Wait for all clients before to launch another communication
        protocol while the batch is not full.

        :param bool get_inputs: If True, compute and return input
        :param bool get_outputs: If True, compute and return output
        :param bool animate: If True, triggers an environment step
        :return:
        """

        client_launched = 0
        # Launch the communication protocol while the batch needs to be filled
        while client_launched < self.batch_size:
            # Run communicate protocol for each client and wait for the last one to finish
            await gather(*[self.__communicate(client=client, client_id=client_id, get_inputs=get_inputs,
                                              get_outputs=get_outputs, animate=animate)
                           for client_id, client in self.clients])
            client_launched += len(self.clients)

    async def __communicate(self, client=None, client_id=None, get_inputs=True, get_outputs=True, animate=True):
        """
        Communication protocol with a client. It goes through different steps:
        1) Eventually send samples to Client
        2) Running steps & Receiving training data
        3) Add data to the Queue

        :param client: TcpIpObject client to communicate with
        :param int client_id: Index of the client
        :param bool get_inputs: If True, compute and return input
        :param bool get_outputs: If True, compute and return output
        :param bool animate: If True, triggers an environment step
        :return:
        """

        loop = get_event_loop()
        if animate:

            # 1) If a sample from Dataset is given, sent it to the TcpIpClient
            if self.batch_from_dataset is not None:
                # Send the sample to the TcpIpClient
                await self.send_command_sample(loop=loop, receiver=client)
                # Pop the first sample of the numpy batch for network in / out
                for field in ['input', 'output']:
                    sample = np.array([])
                    if field in self.batch_from_dataset:
                        # Pop sample from array
                        sample = self.batch_from_dataset[field][0]
                        self.batch_from_dataset[field] = self.batch_from_dataset[field][1:]
                        # Keep the sample in memory
                        self.data_dict[client_id][field] = sample
                    # Send network in / out sample
                    await self.send_data(data_to_send=sample, loop=loop, receiver=client)
                # Pop the first sample of the numpy batch for each additional in / out
                for field in ['dataset_in', 'dataset_out']:
                    # Tell TcpClient if there is additional data for this field
                    await self.send_data(data_to_send=field in self.batch_from_dataset, loop=loop, receiver=client)
                    if field in self.batch_from_dataset:
                        sample = {}
                        # Get each additional data field
                        for key in self.batch_from_dataset[field]:
                            # Pop sample from array
                            sample[key] = self.batch_from_dataset[field][key][0]
                            self.batch_from_dataset[field][key] = self.batch_from_dataset[field][key][1:]
                            # Keep the sample in memory
                            self.data_dict[client_id][field + key] = sample[key]
                        # Send additional in / out sample
                        await self.send_dict(name="additional_data", dict_to_send=sample, loop=loop, receiver=client)

            # 2) Execute n steps, the last one send data computation signal
            visu_dict = {}
            for current_step in range(self.environment_manager.simulations_per_step):
                # Last step is to compute data
                if current_step == self.environment_manager.simulations_per_step - 1:
                    await self.send_command_compute(loop=loop, receiver=client)
                # Otherwise, just a simple step
                else:
                    await self.send_command_step(loop=loop, receiver=client)
                # Receive data while Client has to
                await self.listen_while_not_done(loop=loop, sender=client, data_dict=self.data_dict,
                                                 client_id=client_id)
                # Extract visualization data
                if 'visualisation' in self.data_dict[client_id]:
                    visu_dict[client_id] = self.data_dict[client_id]['visualisation']
            # Update visualization
            if visu_dict != {}:
                self.environment_manager.update_visualizer(visu_dict)

        # 3) Add received data to queue
        data = {}
        for get_data, net_field, add_field in zip([get_inputs, get_outputs], ['input', 'output'],
                                                  ['dataset_in', 'dataset_out']):
            if get_data:
                # Add network field
                data[net_field] = self.data_dict[client_id][net_field]
                # Add each additional dataset
                additional_fields = [key for key in self.data_dict[client_id].keys() if key.__contains__(add_field)]
                data[add_field] = {}
                for field in additional_fields:
                    data[add_field][field[len(add_field):]] = self.data_dict[client_id][field]
        # Add loss data if provided
        if 'loss' in self.data_dict[client_id]:
            data['loss'] = self.data_dict[client_id]['loss']
        # Identify sample
        data['ID'] = client_id
        # Add data to the Queue
        self.data_fifo.put(data)

    def set_dataset_batch(self, batch):
        """
        Receive a batch of data from the Dataset. Samples will be dispatched between clients.

        :param ndarray batch: Batch of data
        :return:
        """

        # Check batch size
        if len(batch['input']) != self.batch_size:
            raise ValueError(f"[{self.name}] The size of batch from Dataset is {len(batch['input'])} while the batch "
                             f"size was set to {self.batch_size}.")
        # Define batch from dataset
        self.batch_from_dataset = batch.copy()

    def close(self):
        """
        Run __close method with asyncio

        :return:
        """

        print(f"[{self.name}] Closing clients...")
        run(self.__close())

    async def __close(self):
        """
        Run server shutdown protocol.

        :return:
        """

        # Send all exit protocol and wait for the last one to finish
        await gather(*[self.__shutdown(client=client, idx=client_id) for client_id, client in self.clients])
        # Close socket
        self.sock.close()

    async def __shutdown(self, client=None, idx=None):
        """
        Send exit command to all clients

        :param client: TcpIpObject client
        :param int idx: Client index
        :return:
        """

        loop = get_event_loop()
        print(f"[{self.name}] Sending exit command to", idx)
        # Send exit command
        await self.send_command_exit(loop=loop, receiver=client)
        await self.send_command_done(loop=loop, receiver=client)
        # Wait for exit confirmation
        data = await self.receive_data(loop=loop, sender=client)
        if data != b'exit':
            raise ValueError(f"Client {idx} was supposed to exit.")

    async def update_visualizer(self, visualization_data, client_id):
        """
        Update Visualizer with visualization data.

        :param dict visualization_data: Dictionary containing visualization data.
        :param int client_id: ID of the TcpIpClient
        :return:
        """

        self.environment_manager.update_visualizer(visualization_data, client_id)

    async def action_on_prediction(self, data, client_id, loop=None, sender=None):
        """
        Define actions to take on a 'prediction' command.

        :param dict data: Dict storing data
        :param int client_id: ID of the TcpIpClient
        :param loop: asyncio.get_event_loop() return
        :param sender: TcpIpObject sender
        :return:
        """

        # Receive network input
        label, network_input = await self.receive_labeled_data(loop=loop, sender=sender)

        # Check that manager hierarchy is well-defined
        if self.environment_manager.data_manager is None:
            raise ValueError("Cannot request prediction if DataManager does not exist")
        elif self.environment_manager.data_manager.manager is None:
            raise ValueError("Cannot request prediction if Manager does not exist")
        elif self.environment_manager.data_manager.manager.network_manager is None:
            raise ValueError("Cannot request prediction if NetworkManager does not exist")

        # Get the prediction from NetworkPrediction
        prediction = self.environment_manager.data_manager.manager.network_manager.compute_online_prediction(
            network_input=network_input[None, ])
        # Send back the prediction to the Client
        await self.send_labeled_data(data_to_send=prediction, label="prediction", receiver=sender,
                                     send_read_command=False)

    async def action_on_visualisation(self, data, client_id, sender, loop):
        """
        Define actions to take on a 'visualization' command.

        :param dict data: Dict storing data
        :param int client_id: ID of the TcpIpClient
        :param loop: asyncio.get_event_loop() return
        :param sender: TcpIpObject sender
        :return:
        """

        # Receive visualization data
        await self.receive_dict(data[client_id], sender=sender, loop=loop)
