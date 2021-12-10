from asyncio import get_event_loop, run, gather
import numpy as np
from queue import SimpleQueue
from DeepPhysX_Core.AsyncSocket.TcpIpObject import TcpIpObject
<<<<<<< HEAD
=======
from DeepPhysX_Core.AsyncSocket.BytesConverter import BytesConverter
>>>>>>> Server and client handle multiple object types and dictionaries
from copy import copy


class TcpIpServer(TcpIpObject):

    def __init__(self,
                 ip_address='localhost',
                 port=10000,
<<<<<<< HEAD
=======
                 data_converter=BytesConverter,
>>>>>>> Server and client handle multiple object types and dictionaries
                 max_client_count=10,
                 batch_size=5,
                 nb_client=5,
                 manager=None):
        """
        TcpIpServer is used to communicate with clients associated with Environment to produce batches for the
        EnvironmentManager.

        :param str ip_address: IP address of the TcpIpObject
        :param int port: Port number of the TcpIpObject
        :param int max_client_count: Maximum number of allowed clients
        :param int batch_size: Number of samples in a batch
        :param int nb_client: Number of expected client connections
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
        print(f"[{self.name}] Waiting for clients:")
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
            print(f"[{self.name}] Client n°{len(self.clients)} connected: {client}")
            self.clients.append(client)

    def initialize(self, param_dict):
        """
        Run __initialize method with asyncio.

        :param dict param_dict: Dictionary of parameters to send to the client's environment
        :return:
        """
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
        for client_id, client in enumerate(self.clients):

            await self.send_dict(name="parameters", dict_to_send=param_dict, loop=loop, receiver=client)

            # Receive visualization data and parameters
            print("SERVER __initialize VISU")
            await self.receive_dict(recv_to=self.data_dict[client_id], sender=client, loop=loop)
            print("SERVER __initialize PARAM")
            await self.receive_dict(recv_to=self.data_dict[client_id], sender=client, loop=loop)
            print("SERVER __initialize END")
        print(self.data_dict)

    def getBatch(self, get_inputs=True, get_outputs=True, animate=True):
        """
        Build a batch from clients samples.

        :param bool get_inputs: If True, compute and return input
        :param bool get_outputs: If True, compute and return output
        :param bool animate: If True, triggers an environment step
        :return: The batch (list of samples), additional data in a dictionary
        """
        # Trigger communication protocol
        run(self.__request_data_from_client(get_inputs=get_inputs, get_outputs=get_outputs, animate=animate))
<<<<<<< HEAD
        data_sorter = {'input': [], 'dataset_in': {}, 'output': [], 'dataset_out': {}, 'loss': []}
=======

        data_sorter = {'input': [], 'output': [], 'loss': []}

>>>>>>> Server and client handle multiple object types and dictionaries
        self.sample_to_client_id = []
        while max(len(data_sorter['input']), len(data_sorter['dataset_in']),
                  len(data_sorter['output']), len(data_sorter['dataset_out']),
                  len(data_sorter['loss']),
                  len(self.sample_to_client_id)) < self.batch_size and not self.data_fifo.empty():
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
        # Launch the communication protocol when a batch needs to be filled
        client_launched = 0
        while client_launched < self.batch_size:
            # Run a communicate protocol for each client and wait for the last one to finish
            await gather(
                *[self.__communicate(client=client, client_id=client_id, get_inputs=get_inputs, get_outputs=get_outputs,
                                     animate=animate)
                  for client_id, client in enumerate(self.clients)])
            client_launched += len(self.clients)

    async def __communicate(self, client=None, client_id=None, get_inputs=True, get_outputs=True, animate=True):
        """
        Communication protocol with a client. It goes trough different steps:
        1) Running steps
        2) Receiving training data
        3) Add data to the Queue

        :param client: TcpIpObject client to communicate with
        :param int client_id: Index of the client
        :param bool get_inputs: If True, compute and return input
        :param bool get_outputs: If True, compute and return output
        :param bool animate: If True, triggers an environment step
        :return:
        """
        loop = get_event_loop()

        # 1) Tell client to compute steps in the environment
        if animate:
            # 1.1) If a sample from Dataset is given, sent it to the Environment
            if self.batch_from_dataset is not None:
                # Send the sample to the TcpIpClient
                await self.send_command_sample(loop=loop, receiver=client)
                # Pop the first sample of the numpy batch for network in / out
                for field in ['input', 'output']:
                    sample = np.array([])
                    if field in self.batch_from_dataset:
                        sample = self.batch_from_dataset[field][0]
                        self.batch_from_dataset[field] = self.batch_from_dataset[field][1:]
                    await self.send_data(data_to_send=sample, loop=loop, receiver=client)
                # Pop the first sample of the numpy batch for each additional in / out
                for field in ['dataset_in', 'dataset_out']:
                    # Is there additional data for this field ?
                    await self.send_data(data_to_send=field in self.batch_from_dataset, loop=loop, receiver=client)
                    if field in self.batch_from_dataset:
                        sample = {}
                        for key in self.batch_from_dataset[field]:
                            sample[key] = self.batch_from_dataset[field][key][0]
                            self.batch_from_dataset[field][key] = self.batch_from_dataset[field][key][1:]
                        await self.send_dict_data(dict_data=sample, loop=loop, receiver=client)

            # 1.2) Execute n steps, the last one send data computation signal
            for current_step in range(self.environment_manager.simulations_per_step):
                if current_step == self.environment_manager.simulations_per_step - 1:
                    await self.send_command_compute(loop=loop, receiver=client)
                else:
                    await self.send_command_step(loop=loop, receiver=client)
<<<<<<< HEAD
            # 2) Receive training data
            await self.listen_while_not_done(loop=loop, sender=client, data_dict=self.data_dict[client_id],
                                             client_id=client_id)
        # 3) Fill the data Queue
        data = {}
        if self.data_dict[client_id]['check']:

            for get_data, data_size, net_field, add_field in zip([get_inputs, get_outputs],
                                                                 [self.in_size, self.out_size],
                                                                 ['input', 'output'],
                                                                 ['dataset_in', 'dataset_out']):
                # Check flat data size
                if get_data and self.data_dict[client_id][net_field].size == data_size.prod():
                    data[net_field] = self.data_dict[client_id][net_field].reshape(data_size)
                    # del self.data_dict[client_id][net_field]
                    # Get additional dataset
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
        else:
            await self.__communicate(client=client, client_id=client_id, get_inputs=get_inputs, get_outputs=get_outputs,
                                     animate=animate)
=======
                # Receive data
                await self.listen_while_not_done(loop=loop, sender=client, data_dict=self.data_dict[client_id], client_id=client_id)

        data = {}
        # Checkin input data size
        if get_inputs:
            data['input'] = self.data_dict[client_id]['input']

        # Checkin output data size
        if get_outputs:
            data['output'] = self.data_dict[client_id]['output']

        if 'loss' in self.data_dict[client_id]:
            data['loss'] = self.data_dict[client_id]['loss']

        data['ID'] = client_id
        self.data_fifo.put(data)
>>>>>>> Server and client handle multiple object types and dictionaries

    def setDatasetBatch(self, batch):
        """
        Receive a batch of data from the Dataset. Samples will be dispatched between clients.

        :param batch: Batch of data
        :return:
        """
        if len(batch['input']) != self.batch_size:
<<<<<<< HEAD
            raise ValueError(f"[{self.name}] The size of batch from Dataset is {len(batch['input'])} while the batch "
                             f"size was set to {self.batch_size}.")
=======
            return ValueError(
                f"[TcpIpServer] The size of batch from Dataset is {len(batch['input'])} while the batch size"
                f"was set to {self.batch_size}.")
>>>>>>> Server and client handle multiple object types and dictionaries
        self.batch_from_dataset = copy(batch)

    def applyPrediction(self, prediction):
        """
        Run __applyPrediction method with asyncio

        :param list prediction: Batch of prediction data
        :return:
        """
        run(self.__applyPrediction(prediction))

    async def __applyPrediction(self, prediction):
        """
        Share out the prediction tensors between the corresponding clients.

        :param list prediction: Batch of prediction data
        :return:
        """
        loop = get_event_loop()
        # # Check the prediction batch size
        # if len(prediction) != self.batch_size:
        #     raise ValueError(f"[TcpIpServer] The length of the prediction batch mismatch the expected batch size.")
        # Send each prediction data to a client
        for client_id, data in enumerate(prediction):
            # Tell the client to receive and apply prediction
            await self.send_command_prediction(loop=loop, receiver=self.clients[self.sample_to_client_id[client_id]])
            # Send prediction data to the client
            await self.send_data(data_to_send=np.array(data, dtype=float), loop=loop,
                                 receiver=self.clients[self.sample_to_client_id[client_id]])

    def close(self):
        """
        Run __close method with asyncio

        :return:
        """
        run(self.__close())

    async def __close(self):
        """
        Run server shutdown protocol.

        :return:
        """
        # Send all exit protocol and wait for the last one to finish
        await gather(
            *[self.__shutdown(client=client, idx=client_id) for client_id, client in enumerate(self.clients)])
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
        await self.send_command_exit(loop=loop, receiver=client)
        # Wait for exit confirmation
        data = await self.receive_data(loop=loop, sender=client)
        if data != b'exit':
            raise ValueError(f"[{self.name}] Client {idx} was supposed to exit.")

<<<<<<< HEAD
    async def listen_while_not_done(self, loop, sender, data_dict, client_id=None):
        """
        Listening to data until command 'done' is received.

        :param loop: asyncio.get_event_loop() return
        :param sender: TcpIpObject sender
        :param data_dict: Dictionary in which data is stored
        :param client_id: ID of the sender client
        :return:
        """
        # Execute while command 'done' is not received
        while (cmd := await self.receive_data(loop=loop, sender=sender)) != self.command_dict['done']:
            # With command visualization, receive each field of the visualization dict
            if cmd == self.command_dict['visualization']:
                visu_dict = {}
                # Receive fields until 'done' command
                await self.listen_while_not_done(loop=loop, sender=sender, data_dict=visu_dict, client_id=client_id)
                # Update the visualization
                await self.update_visualizer(visu_dict, client_id)
            # Simply get labeled data once for other commands
            else:
                label, param = await self.receive_labeled_data(loop=loop, sender=sender)
                data_dict[label] = param
                # If command prediction is received then compute and send back the network prediction
                if cmd == self.command_dict['prediction']:
                    await self.compute_and_send_prediction(network_input=data_dict[label], receiver=sender)

=======
>>>>>>> Server and client handle multiple object types and dictionaries
    async def compute_and_send_prediction(self, network_input, receiver):
        """
        Compute and send back the network prediction.

        :param network_input: Input of the network
        :param receiver: TcpIpObject receiver
        :return:
        """
        # Check that managers can communicate
        if self.environment_manager.data_manager is None:
            raise ValueError("Cannot request prediction if DataManager does not exist")
        elif self.environment_manager.data_manager.manager is None:
            raise ValueError("Cannot request prediction if Manager does not exist")
        elif self.environment_manager.data_manager.manager.network_manager is None:
            raise ValueError("Cannot request prediction if NetworkManager does not exist")
<<<<<<< HEAD
        # Get the prediction of the network
        prediction = self.environment_manager.requestPrediction(network_input=network_input[None, ])
        # Send the prediction back to the TcpIpClient
        await self.send_labeled_data(data_to_send=prediction, label="prediction", receiver=receiver,
                                     send_read_command=False)
=======
        else:
            prediction = self.environment_manager.data_manager.manager.network_manager.computeOnlinePrediction(
                network_input=network_input[None,])
            await self.send_labeled_data(data_to_send=prediction, label="prediction", receiver=receiver, send_read_command=False)
>>>>>>> Server and client handle multiple object types and dictionaries

    async def update_visualizer(self, visualization_data, client_id):
        """
        Send the updated visualization data to the associate manager.

        :param visualization_data: Dict containing the data fields to update the visualization
        :param client_id: ID of the client
        :return:
        """
        self.environment_manager.updateVisualizer(visualization_data, client_id)

    async def action_on_prediction(self, data, client_id, sender=None, loop=None):
        label, param = await self.receive_labeled_data(loop=loop, sender=sender)
        await self.compute_and_send_prediction(network_input=param, receiver=sender)
