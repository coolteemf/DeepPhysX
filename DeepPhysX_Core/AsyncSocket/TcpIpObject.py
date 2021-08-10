import socket

from DeepPhysX_Core.AsyncSocket.BytesBaseConverter import BytesBaseConverter


class TcpIpObject:

    def __init__(self, ip_address='localhost', port=10000, data_converter=BytesBaseConverter):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.ip_address = ip_address
        self.port = port
        self.data_converter = data_converter()

    async def run(self):
        pass

    async def communicate(self, client=None, server=None, idx=None):
        pass

    def bytes_size(self, data_as_bytes):
        return len(data_as_bytes)

    async def send_data(self, data_to_send, loop, receiver):
        # Cast data to bytes field
        data_as_bytes = self.data_converter.data_to_bytes(data_to_send)

        # Size of tha data to send
        data_size = self.bytes_size(data_as_bytes)

        # Send the size of the next receive
        await loop.sock_sendall(sock=receiver, data=data_size.to_bytes(4, byteorder='big'))

        # Send the actual data
        await loop.sock_sendall(sock=receiver, data=data_as_bytes)

    async def receive_data(self, loop, sender, expect_command=False, raw_bytes=False):
        # Maximum read sizes array
        read_sizes = [4096, 2048, 1024, 512, 256]
        read_size_idx = 0

        # Always expect to receive the size of the data to read first
        data_size_to_read = int.from_bytes(await loop.sock_recv(sender, 4), 'big')
        data_as_bytes = b''

        # Proceed to read chunk by chunk
        while data_size_to_read > 0:

            # Select the good amount of bytes to read
            while read_size_idx < len(read_sizes) and data_size_to_read < read_sizes[read_size_idx]:
                read_size_idx += 1

            # If the amount of bytes to read is too small then read it all
            chunk_size_to_read = data_size_to_read if read_size_idx >= len(read_sizes) else read_sizes[read_size_idx]

            # Try to read at most "chunk_size_to_read" bytes from the socket
            data_received_as_bytes = await loop.sock_recv(sender, chunk_size_to_read)
            # Todo: add security with <<await asyncio.wait_for(loop.sock_recv(sender, chunk_size_to_read), timeout=1.)>>

            # Accumulate the data
            data_as_bytes += data_received_as_bytes
            data_size_to_read -= self.bytes_size(data_received_as_bytes)

        # Return the data in the expected format
        return data_as_bytes if expect_command or raw_bytes else self.data_converter.bytes_to_data(data_as_bytes)

    async def send_command(self,  loop, receiver, command=''):
        command_dict = {'exit': b'exit', 'wait': b'wait', 'step': b'step'}
        try:
            cmd = command_dict[command]
        except KeyError:
            raise KeyError(f"\"{command}\" is not a valid command. use {command_dict.keys()} instead")

        # Cast data to bytes field
        data_as_bytes = cmd

        # Size of tha data to send
        data_size = self.bytes_size(data_as_bytes)

        # Send the size of the next receive
        await loop.sock_sendall(sock=receiver, data=data_size.to_bytes(4, byteorder='big'))

        # Send the actual data
        await loop.sock_sendall(sock=receiver, data=data_as_bytes)

    async def send_exit_command(self, loop, receiver):
        await self.send_command(loop=loop, receiver=receiver, command='exit')

    async def send_wait_command(self, loop, receiver):
        await self.send_command(loop=loop, receiver=receiver, command='wait')

    async def send_step_command(self, loop, receiver):
        await self.send_command(loop=loop, receiver=receiver, command='step')
