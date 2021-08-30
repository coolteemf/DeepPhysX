import asyncio
import numpy as np
from queue import SimpleQueue
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
        self.data_fifo = SimpleQueue()
        self.in_size = None
        self.out_size = None
        self.data_dict = []
        # Reference to EnvironmentManager
        self.environmentManager = None

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
        self.data_dict = {client_ID: {} for client_ID in range(len(self.clients))}

        # Initialisation process for each client
        for client_id, client in enumerate(self.clients):

            # Send parameters
            for key in param_dict:
                await self.send_command_read(loop=loop, receiver=client)
                # Send the parameter (label + data)
                await self.send_labeled_data(data_to_send=param_dict[key], label=key, loop=loop, receiver=client)
            # Tell the client to stop receiving data
            await self.send_command_done(loop=loop, receiver=client)

            # Receive parameters
            await self.listen_while_not_done(loop=loop, sender=client, data_dict=self.data_dict[client_id])

            # if 'addvedo' in self.data_dict[client_id] and self.data_dict[client_id]['addvedo']:
            #
            #     # Position typo check
            #     positions = self.data_dict[client_id]['positions'] if 'positions' in self.data_dict[client_id] else np.array([])
            #     positions = self.data_dict[client_id]['position'] if 'position' in self.data_dict[client_id] else pos
            #
            #     # cell existence/typo check
            #     cells = self.data_dict[client_id]['cells'] if 'cells' in self.data_dict[client_id] else None
            #     cells = self.data_dict[client_id]['cell'] if 'cell' in self.data_dict[client_id] else pos
            #
            #     self.environmentManager.visualizer.addObject(positions=positions, cells=cells)
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
        inputs = []
        outputs = []
        while not self.data_fifo.empty():
            data = self.data_fifo.get()
            if len(data) == 2:
                inputs.append(data[0])
                outputs.append(data[1])
            elif len(data) == 1:
                if get_inputs:
                    inputs.append(data[0])
                else:
                    outputs.append(data[0])
            else:
                raise ValueError("Empty data given to fifo, cannot generate batch")
        return [inputs, outputs], self.data_dict

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
        while self.data_fifo.qsize() < self.batch_size:
            # Run a communicate protocol for each client and wait for the last one to finish
            await asyncio.gather(
                *[self.__communicate(client=client, client_id=client_id, get_inputs=get_inputs, get_outputs=get_outputs,
                                     animate=animate)
                  for client_id, client in enumerate(self.clients)])

    async def __communicate(self, client=None, client_id=None, get_inputs=True, get_outputs=True, animate=True):
        """
        Communication protocol with a client. It goes trough different steps: 1) Running steps 2) Receiving additional
        data 3) Compute IO data 4) Check the IO sample 5) Receive the IO data 6) Add sample to batch 7) If sample is
        not exploitable save the wrong sample

        :param client: TcpIpObject client to communicate with
        :param int client_id: Index of the client
        :param bool get_inputs: If True, compute and return input
        :param bool get_outputs: If True, compute and return output
        :param bool animate: If True, triggers an environment step
        :return:
        """
        loop = asyncio.get_event_loop()

        # 1) Tell client to compute steps in the environment
        if animate:
            # Execute n steps, the last one send data computation signal
            for current_step in range(self.environmentManager.simulations_per_step):
                if current_step == self.environmentManager.simulations_per_step - 1:
                    await self.send_command_compute(loop=loop, receiver=client)
                else:
                    await self.send_command_step(loop=loop, receiver=client)
                # Receive parameters
                await self.listen_while_not_done(loop=loop, sender=client, data_dict=self.data_dict[client_id])

        # If the sample is exploitable
        if 'check' in self.data_dict[client_id] and self.data_dict[client_id]['check']:
            data = []
            # Checkin input data size
            if get_inputs and self.data_dict[client_id]['input'].size == self.in_size.prod():
                data.append(self.data_dict[client_id]['input'].reshape(self.in_size))

            # Checkin output data size
            if get_outputs and self.data_dict[client_id]['output'].size == self.out_size.prod():
                data.append(self.data_dict[client_id]['output'].reshape(self.out_size))
            if len(data) > 0:
                self.data_fifo.put(data)
        # If the sample is wrong
        else:
            # 7) record wrong sample
            pass

    def applyPrediction(self, prediction):
        """
        Run __applyPrediction method with asyncio

        :param list prediction: Batch of prediction data
        :return:
        """
        asyncio.run(self.__applyPrediction(prediction))

    async def __applyPrediction(self, prediction):
        """
        Share out the prediction tensors between the corresponding clients.

        :param list prediction: Batch of prediction data
        :return:
        """
        loop = asyncio.get_event_loop()
        # # Check the prediction batch size
        # if len(prediction) != self.batch_size:
        #     raise ValueError(f"[TcpIpServer] The length of the prediction batch mismatch the expected batch size.")
        # Send each prediction data to a client
        for client_id, data in enumerate(prediction):
            client_id = client_id % len(self.clients)
            # Tell the client to receive and apply prediction
            await self.send_command_prediction(loop=loop, receiver=self.clients[client_id])
            # Send prediction data to the client
            await self.send_data(data_to_send=np.array(data, dtype=float), loop=loop, receiver=self.clients[client_id])

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
