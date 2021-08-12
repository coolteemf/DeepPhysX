import asyncio

from DeepPhysX_Core.AsyncSocket.TcpIpObject import TcpIpObject, BytesNumpyConverter


class TcpIpServer(TcpIpObject):

    def __init__(self, ip_address='localhost', port=10000, data_converter=BytesNumpyConverter,
                 max_client_count=10, batch_size=5, nb_client=5):
        """
        TcpIpServer is used to communicate with clients associated with Environment to produce batches for the
        EnvironmentManager.

        :param str ip_address: IP address of the TcpIpObject
        :param int port: Port number of the TcpIpObject
        :param data_converter: BytesBaseConverter class to convert data to bytes (NumPy by default)
        :param int max_client_count: Maximum number of allowed clients
        :param int batch_size: Number of samples in a batch
        :param int nb_client: Number of expected client connections
        """
        super(TcpIpServer, self).__init__(ip_address=ip_address, port=port, data_converter=data_converter)
        # Bind to server address
        print(f'Binding to IP Address: {ip_address} on PORT: {port} with maximum client count: {max_client_count}')
        self.sock.bind((ip_address, port))
        self.sock.listen(max_client_count)
        self.sock.setblocking(False)
        # Expect a defined number of clients
        self.clients = []
        self.nb_client = min(nb_client, max_client_count)
        # Init data to communicate with EnvironmentManager and Clients
        self.batch_size = batch_size
        self.current_batch = [[], []]
        self.next_batch = [[], []]
        self.in_size = None
        self.out_size = None
        self.data_dict = {}
        # Reference to EnvironmentManager
        self.manager = None

    def connect(self):
        """
        Run __connect method with asyncio.

        :return:
        """
        print(f"Waiting for clients:")
        asyncio.run(self.__connect())

    async def __connect(self):
        """
        Accept connections from clients.

        :return:
        """
        loop = asyncio.get_event_loop()
        # Accept clients connections one by one
        for _ in range(self.nb_client):
            client, _ = await loop.sock_accept(self.sock)
            print(f'Client n°{len(self.clients)} connected: {client}')
            self.clients.append(client)

    def initialize(self, param_dict):
        """
        Run __initialize method with asyncio.

        :param dict param_dict: Dictionary of parameters to send to the client's environment
        :return:
        """
        asyncio.run(self.__initialize(param_dict))

    async def __initialize(self, param_dict):
        """
        Send parameters to the clients to create their environments, receive parameters from clients in exchange.

        :param dict param_dict: Dictionary of parameters to send to the client's environment
        :return: Dictionary of parameters for each environment to send the manager
        """
        loop = asyncio.get_event_loop()

        # Empty dictionaries for received parameters from clients
        env_param_dicts = [{} for _ in range(len(self.clients))]

        # Initialisation process for each client
        for client_id, client in enumerate(self.clients):

            # Send parameters
            for key in param_dict:
                # Prepare the client to receive data
                await self.send_command_receive(loop=loop, receiver=client)
                # Send the parameter (label + data)
                await self.send_labeled_data(data_to_send=param_dict[key], label=key, loop=loop, receiver=client)
            # Tell the client to stop receiving data
            await self.send_command_done(loop=loop, receiver=client)

            # Receive parameters
            cmd = b''
            # Receive data while the client did not say to stop
            while cmd != b'done':
                # Receive and check client command
                cmd = await self.receive_data(loop=loop, sender=client, is_bytes_data=True)
                if cmd not in [b'done', b'recv']:
                    raise ValueError(f"Unknown command {cmd}, must be in {[b'done', b'recv']}")
                # Receive data if the client did not say to stop
                if cmd != b'done':
                    label, param = await self.receive_labeled_data(loop=loop, sender=client)
                    env_param_dicts[client_id][label] = param

            # Get data sizes from the first client (these sizes are equals in all environments)
            if client_id == 0:
                # Ask the client to send data sizes
                await self.send_command_size(loop=loop, receiver=client)
                # Receive input size the output size
                in_size = await self.receive_data(loop=loop, sender=client)
                out_size = await self.receive_data(loop=loop, sender=client)
                # Convert array from float to int
                self.in_size = in_size.astype(int)
                self.out_size = out_size.astype(int)
            else:
                # Tell the other clients not to send data sizes
                await self.send_command_done(loop=loop, receiver=client)

        return env_param_dicts

    def getBatch(self, get_inputs=True, get_outputs=True, animate=True):
        """
        Build a batch from clients samples.

        :param bool get_inputs: If True, compute and return input
        :param bool get_outputs: If True, compute and return output
        :param bool animate: If True, triggers an environment step
        :return: The batch (list of samples), additional data in a dictionary
        """
        # Trigger communication protocol
        asyncio.run(self.__run(get_inputs=get_inputs, get_outputs=get_outputs, animate=animate))
        # Get current batch, re-initialize next batch
        batch = self.current_batch.copy()
        self.current_batch = [self.next_batch[0][:self.batch_size], self.next_batch[1][:self.batch_size]]
        self.next_batch = [self.next_batch[0][self.batch_size:], self.next_batch[1][self.batch_size:]]
        # Return batch to environment manager
        return batch, self.data_dict

    async def __run(self, get_inputs=True, get_outputs=True, animate=True):
        """
        Trigger a communication protocol for each client. Wait for all clients before to launch another communication
        protocol while the batch is not full.

        :param bool get_inputs: If True, compute and return input
        :param bool get_outputs: If True, compute and return output
        :param bool animate: If True, triggers an environment step
        :return:
        """
        # Launch the communication protocol when a batch needs to be filled
        while len(self.current_batch[0]) < self.batch_size:
            # Run a communicate protocol for each client and wait for the last one to finish
            await asyncio.gather(
                *[self.__communicate(client=client, idx=client_id, get_inputs=get_inputs, get_outputs=get_outputs,
                                     animate=animate)
                  for client_id, client in enumerate(self.clients)])

    async def __communicate(self, client=None, idx=None, get_inputs=True, get_outputs=True, animate=True):
        """
        Communication protocol with a client. It goes trough different steps: 1) Running steps 2) Receiving additional
        data 3) Compute IO data 4) Check the IO sample 5) Receive the IO data 6) Add sample to batch 7) If sample is
        not exploitable save the wrong sample

        :param client: TcpIpObject client to communicate with
        :param int idx: Index of the client
        :param bool get_inputs: If True, compute and return input
        :param bool get_outputs: If True, compute and return output
        :param bool animate: If True, triggers an environment step
        :return:
        """
        loop = asyncio.get_event_loop()

        # 1) Tell client to compute steps in the environment
        if animate:
            for _ in range(self.manager.simulations_per_step):
                await self.send_command_step(loop=loop, receiver=client)

        # 2) Receive whatever data from the client's environment
        # Todo: add this loop when a b'done' is send at the end of the time step
        # cmd = b''
        # while cmd != b'done':
        #     cmd = await self.receive_data(loop=loop, sender=client, is_bytes_data=True)
        #     if cmd not in [b'done', b'recv']:
        #         raise ValueError(f"Unknown command {cmd}, must be in {[b'done', b'recv']}")
        #     name, data = await self.receive_named_data(loop=loop, sender=client)
        #     self.data_dict[name].append(data)

        # 3) Tell client to compute data in the environment
        if get_inputs:
            await self.send_command_compute_input(loop=loop, receiver=client)
        if get_outputs:
            await self.send_command_compute_output(loop=loop, receiver=client)

        # 4) Tell the client to check data sample
        await self.send_command_check(loop=loop, receiver=client)
        check = bool(await self.receive_data(loop=loop, sender=client, is_bytes_data=True))

        # If the sample is exploitable
        if check:
            data_in, data_out = None, None

            # 5.1) Tell the client to send the input data
            if get_inputs:
                await self.send_command_get_input(loop=loop, receiver=client)
                data_in = await self.receive_data(loop=loop, sender=client)
                # Checkin input data size
                if not data_in.size == self.in_size.prod():
                    data_in = None

            # 5.2) Tell the client to send the output data
            if get_outputs:
                await self.send_command_get_output(loop=loop, receiver=client)
                data_out = await self.receive_data(loop=loop, sender=client)
                # Checkin output data size
                if not data_out.size == self.out_size.prod():
                    data_out = None

            # 6) Add data to batch
            if not(get_inputs and data_in is None) and not(get_outputs and data_out is None):
                self.manageBatch(data_in, data_out)

        # If the sample is wrong
        else:
            # 7) record wrong sample
            pass

    def manageBatch(self, data_in, data_out):
        """
        Add IO data to the current batch.

        :param data_in: Input data
        :param data_out: Output data
        :return:
        """
        # Reshape data which was flatten when sent on socket
        if data_in is not None:
            data_in = data_in.reshape(self.in_size)
        if data_out is not None:
            data_out = data_out.reshape(self.out_size)
        # If the current batch in not filled, add data to current batch
        if len(self.current_batch[0]) < self.batch_size:
            self.current_batch[0].append(data_in)
            self.current_batch[1].append(data_out)
        # If the current batch is already filled, add data to the next one
        else:
            self.next_batch[0].append(data_in)
            self.next_batch[1].append(data_out)

    def close(self):
        """
        Run __close method with asyncio

        :return:
        """
        asyncio.run(self.__close())

    async def __close(self):
        """
        Run server shutdown protocol.

        :return:
        """
        # Send all exit protocol and wait for the last one to finish
        await asyncio.gather(
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
        loop = asyncio.get_event_loop()
        print("Sending exit command to", idx)
        await self.send_command_exit(loop=loop, receiver=client)
        # Wait for exit confirmation
        data = await self.receive_data(loop=loop, sender=client, is_bytes_data=True)
        if data != b'exit':
            raise ValueError(f"Client {idx} was supposed to exit.")
