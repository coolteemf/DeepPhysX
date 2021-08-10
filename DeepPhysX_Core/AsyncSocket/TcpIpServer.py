import asyncio

from DeepPhysX_Core.AsyncSocket.TcpIpObject import TcpIpObject, BytesBaseConverter


class TcpIpServer(TcpIpObject):

    def __init__(self, ip_address='localhost', port=10000, data_converter=BytesBaseConverter,
                 max_client_count=10, nb_batches=5, batch_size=5, nb_client=5):
        super(TcpIpServer, self).__init__(ip_address=ip_address, port=port, data_converter=data_converter)
        print(f'Binding to IP Address : {ip_address} on PORT : {port}')
        self.sock.bind((ip_address, port))

        print(f"Maximum client count : {max_client_count}")
        self.sock.listen(max_client_count)
        self.sock.setblocking(False)

        self.clients = []
        self.nb_client = nb_client if nb_client < max_client_count else max_client_count

        self.nb_batch = nb_batches
        self.batch_size = batch_size
        self.current_batch = []
        self.id_batch = 0

    def start_server(self):
        print(f"Waiting for clients:")
        asyncio.run(self.run())
        self.sock.close()

    async def run(self):
        self.loop = asyncio.get_event_loop()
        # while not self.exit_condition():
        for _ in range(self.nb_client):
            client, _ = await self.loop.sock_accept(self.sock)
            print(f'Client n°{len(self.clients)} connected:')  # {client}')
            self.clients.append(client)

        while not self.exit_condition():
            # Send all communicate protocol and wait for the last one to finish
            await asyncio.gather(*[self.communicate(client=client, idx=client_id) for client_id, client in enumerate(self.clients)])

        # Send all exit protocol and wait for the last one to finish
        await asyncio.gather(*[self.exit_protocol(client=client, idx=client_id) for client_id, client in enumerate(self.clients)])

        self.sock.close()

    async def communicate(self, client=None, sock=None, idx=None):
        loop = asyncio.get_event_loop()

        # Tell Client to generate and send a data
        await self.send_step_command(loop=loop, receiver=client)

        # Waiting for the data to arrive
        data = await self.receive_data(loop=loop, sender=client)

        # Sending back the received data
        await self.send_data(data_to_send=data, loop=loop, receiver=client)

        # Validate data from client and add to batch
        self.manage_batch(data)

    async def exit_protocol(self, client=None, sock=None, idx=None):
        loop = asyncio.get_event_loop()
        print("Sending exit command to", idx)
        await self.send_exit_command(loop=loop, receiver=client)
        # Wait for exit confirmation
        data = await self.receive_data(loop=loop, sender=client, expect_command=True)
        if data != b'exit':
            raise ValueError(f"Client {idx} was supposed to exit.")

    def manage_batch(self, tensor):
        self.current_batch.append(tensor)
        # If the batch is complete, reset current batch and set all clients to busy
        if len(self.current_batch) == self.batch_size:
            self.id_batch += 1
            print(f"Batch n°{self.id_batch} with lenght {len(self.current_batch)}/{self.batch_size}")
            self.current_batch = []

    def exit_condition(self):
        # print("Try to exit with values", self.id_batch, self.nb_batch, self.id_batch >= self.nb_batch)
        return self.id_batch >= self.nb_batch

    async def manage_timeout_error(self, loop, sender):
        print("Closing client prematurement")
        await self.send_exit_command(loop=loop, receiver=sender)
        self.clients.remove(sender)
