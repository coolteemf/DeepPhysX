import asyncio
import copy

from DeepPhysX_Core.AsyncSocket.TcpIpObject import TcpIpObject, BytesNumpyConverter


class TcpIpServer(TcpIpObject):

    def __init__(self, ip_address='localhost', port=10000, data_converter=BytesNumpyConverter,
                 max_client_count=10, batch_size=5, nb_client=5):
        """

        :param ip_address:
        :param port:
        :param data_converter:
        :param max_client_count:
        :param batch_size:
        :param nb_client:
        """
        super(TcpIpServer, self).__init__(ip_address=ip_address, port=port, data_converter=data_converter)
        print(f'Binding to IP Address: {ip_address} on PORT: {port} with maximum client count: {max_client_count}')
        self.sock.bind((ip_address, port))
        self.sock.listen(max_client_count)
        self.sock.setblocking(False)

        self.clients = []
        self.nb_client = min(nb_client, max_client_count)

        self.batch_size = batch_size
        self.current_batch = [[], []]
        self.next_batch = [[], []]
        self.in_size = None
        self.out_size = None

        self.manager = None

    def start_server(self):
        """

        :return:
        """
        print(f"Waiting for clients:")
        asyncio.run(self.connect_clients())

    async def connect_clients(self):
        """

        :return:
        """
        loop = asyncio.get_event_loop()
        # Accept clients connections one by one
        for _ in range(self.nb_client):
            client, _ = await loop.sock_accept(self.sock)
            print(f'Client n°{len(self.clients)} connected: {client}')
            self.clients.append(client)

    async def create_data(self, get_inputs, get_outputs):
        """

        :param get_inputs:
        :param get_outputs:
        :return:
        """
        # Launch the communication protocol when a batch needs to be filled
        while len(self.current_batch[0]) < self.batch_size:
            # Send all communicate protocol and wait for the last one to finish
            await asyncio.gather(
                *[self.communicate(client=client, idx=client_id, get_input=get_inputs, get_output=get_outputs)
                  for client_id, client in enumerate(self.clients)])

    async def shutdown(self):
        """

        :return:
        """
        # Send all exit protocol and wait for the last one to finish
        await asyncio.gather(
            *[self.exit_protocol(client=client, idx=client_id) for client_id, client in enumerate(self.clients)])
        self.sock.close()

    async def communicate(self, client=None, sock=None, idx=None, get_input=True, get_output=True):
        """

        :param client:
        :param sock:
        :param idx:
        :param get_input:
        :param get_output:
        :return:
        """
        loop = asyncio.get_event_loop()

        # Tell the client to send the sizes of data (np.array will be flatten when sent on socket)
        if self.in_size is None or self.out_size is None:
            await self.send_size_command(loop=loop, receiver=client)
            in_size = await self.receive_data(loop=loop, sender=client)
            out_size = await self.receive_data(loop=loop, sender=client)
            self.in_size = in_size.astype(int)
            self.out_size = out_size.astype(int)

        # Tell client to compute steps in the environment
        for _ in range(self.manager.simulations_per_step):
            await self.send_step_command(loop=loop, receiver=client)

        # Tell client to compute data in the environment
        if get_input:
            await self.send_compute_input_command(loop=loop, receiver=client)
        if get_output:
            await self.send_compute_output_command(loop=loop, receiver=client)

        # Tell the client to check data sample
        await self.send_check_command(loop=loop, receiver=client)
        check = bool(await self.receive_data(loop=loop, sender=client, expect_command=True))

        # If the sample is exploitable
        if check:
            data_in, data_out = None, None

            # Tell the client to send the input data
            if get_input:
                await self.send_get_input_command(loop=loop, receiver=client)
                data_in = await self.receive_data(loop=loop, sender=client)
                # Checkin input data size
                if not data_in.size == self.in_size.prod():
                    data_in = None

            # Tell the client to send the output data
            if get_output:
                await self.send_get_output_command(loop=loop, receiver=client)
                data_out = await self.receive_data(loop=loop, sender=client)
                # Checkin output data size
                if not data_out.size == self.out_size.prod():
                    data_out = None

            # Add data to batch
            if not(get_input and data_in is None) and not(get_output and data_out is None):
                self.manage_batch(data_in, data_out)

        # If the sample is wrong
        else:
            # record wrong sample
            pass

    async def exit_protocol(self, client=None, idx=None):
        """

        :param client:
        :param idx:
        :return:
        """
        loop = asyncio.get_event_loop()
        print("Sending exit command to", idx)
        await self.send_exit_command(loop=loop, receiver=client)
        # Wait for exit confirmation
        data = await self.receive_data(loop=loop, sender=client, expect_command=True)
        if data != b'exit':
            raise ValueError(f"Client {idx} was supposed to exit.")

    def manage_batch(self, data_in, data_out):
        """

        :param data_in:
        :param data_out:
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

    def getBatch(self, get_inputs, get_outputs):
        """

        :param get_inputs:
        :param get_outputs:
        :return:
        """
        # Start batch creation
        asyncio.run(self.create_data(get_inputs, get_outputs))
        # Get current batch, re-initialize next batch
        batch = copy.copy(self.current_batch)
        self.current_batch = [self.next_batch[0][:self.batch_size], self.next_batch[1][:self.batch_size]]
        self.next_batch = [self.next_batch[0][self.batch_size:], self.next_batch[1][self.batch_size:]]
        # Return batch to environment manager
        return batch

    def close(self):
        """

        :return:
        """
        asyncio.run(self.shutdown())
