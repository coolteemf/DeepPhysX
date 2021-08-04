import asyncio
import socket
import sys

import numpy as np


class TcpIpServerAsync:

    def __init__(self, ip_address='localhost', port=10000, max_client_count=10, nb_batches=5, batch_size=5,
                 nb_client=5):
        print(f"Server Creation")
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print(f'Binding to IP Address : {ip_address} on PORT : {port}')
        self.server.bind((ip_address, port))
        print(f"Maximum client count : {max_client_count}")
        self.server.listen(max_client_count)
        self.server.setblocking(False)
        self.clients = []
        self.busy_client = []
        self.nb_client = nb_client

        self.nb_batch = nb_batches
        self.batch_size = batch_size
        self.current_batch = []
        self.id_batch = 0
        self.mainloop = None
        print("#####################################################")
        print(f"Waiting for clients:")
        asyncio.run(self.run_server())
        self.server.close()
        print("End init server")

    async def run_server(self):
        self.mainloop = asyncio.get_event_loop()
        # while not self.exit_condition():
        for _ in range(self.nb_client):
            client, _ = await self.mainloop.sock_accept(self.server)
            print(f'Client n°{len(self.clients)} connected: {client}')
            self.clients.append(client)
            self.busy_client.append(True)
            self.mainloop.create_task(self.handle_client(client, len(self.clients)-1))
        print()
        while not self.exit_condition():
            await asyncio.sleep(1)
        self.server.close()

    async def handle_client(self, client, idx):
        loop = asyncio.get_event_loop()
        try:
            while not self.exit_condition():
                if self.busy_client[idx]:
                    await loop.sock_sendall(client, bytes('step', 'utf-8'))
                    tensor_recv = np.frombuffer(await loop.sock_recv(client, 255))
                    tensor = tensor_recv.tobytes()
                    print('Server send back to client', idx, tensor_recv)
                    await loop.sock_sendall(client, tensor)
                    self.busy_client[idx] = False
                    self.manage_batch(tensor_recv)
                else:
                    await loop.sock_sendall(client, bytes('wait', 'utf-8'))
                    data = await loop.sock_recv(client, 5)
                    if data != b'wait':
                        print(idx, "Should be 'wait' but received", str(data))
                        raise ValueError(f"Client {idx} should be waiting.")
            print("Sending exit command to", idx)
            await loop.sock_sendall(client, bytes('exit', 'utf-8'))
            data = await loop.sock_recv(client, 5)
            if data != b'exit':
                raise ValueError(f"Client {idx} was suppose to exit.")
        except KeyboardInterrupt:
            print("KEYBOARD INTERRUPT: CLOSING PROCEDURE")
        finally:
            print("Closing socket", idx)
            self.server.close()

    def manage_batch(self, tensor):
        self.current_batch.append(tensor)
        # If the batch is complete, reset current batch and set all clients to busy
        if len(self.current_batch) == self.batch_size:
            self.id_batch += 1
            print(self.id_batch, self.current_batch)
            self.current_batch = []
            self.busy_client = [True for _ in range(len(self.clients))]
            print()
        # If the batch is not complete but there is no busy clients, set them all to busy
        if not (True in self.busy_client):
            self.busy_client = [True for _ in range(len(self.clients))]

    def exit_condition(self):
        # print("Try to exit with values", self.id_batch, self.nb_batch, self.id_batch >= self.nb_batch)
        return self.id_batch >= self.nb_batch
