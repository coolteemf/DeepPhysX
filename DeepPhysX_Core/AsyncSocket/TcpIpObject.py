import socket

from DeepPhysX_Core.AsyncSocket.BytesNumpyConverter import BytesNumpyConverter


class TcpIpObject:

    def __init__(self, ip_address='localhost', port=10000, data_converter=BytesNumpyConverter):
        """

        :param ip_address:
        :param port:
        :param data_converter:
        """
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.ip_address = ip_address
        self.port = port
        self.data_converter = data_converter()

    async def communicate(self, client=None, server=None, idx=None):
        """

        :param client:
        :param server:
        :param idx:
        :return:
        """
        pass

    def bytes_size(self, data_as_bytes):
        """

        :param data_as_bytes:
        :return:
        """
        return len(data_as_bytes)

    async def send_data(self, data_to_send, loop, receiver, do_convert=True):
        """

        :param data_to_send:
        :param loop:
        :param receiver:
        :param do_convert:
        :return:
        """
        # Cast data to bytes field
        data_as_bytes = self.data_converter.data_to_bytes(data_to_send) if do_convert else data_to_send
        # Size of tha data to send
        data_size = self.bytes_size(data_as_bytes)
        # Send the size of the next receive
        await loop.sock_sendall(sock=receiver, data=data_size.to_bytes(4, byteorder='big'))
        # Send the actual data
        await loop.sock_sendall(sock=receiver, data=data_as_bytes)

    async def receive_data(self, loop, sender, expect_command=False, raw_bytes=False):
        """

        :param loop:
        :param sender:
        :param expect_command:
        :param raw_bytes:
        :return:
        """
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

    async def send_command(self, loop, receiver, command=''):
        """

        :param loop:
        :param receiver:
        :param command:
        :return:
        """
        # Check if the command exists
        command_dict = {'exit': b'exit', 'step': b'step', 'check': b'test', 'size': b'size',
                        'compute_in': b'c_in', 'compute_out': b'c_ou', 'get_in': b'g_in', 'get_out': b'g_ou'}
        try:
            cmd = command_dict[command]
        except KeyError:
            raise KeyError(f"\"{command}\" is not a valid command. Use {command_dict.keys()} instead.")
        # Send command as a byte data
        await self.send_data(data_to_send=cmd, loop=loop, receiver=receiver, do_convert=False)

    async def send_exit_command(self, loop, receiver):
        await self.send_command(loop=loop, receiver=receiver, command='exit')

    async def send_step_command(self, loop, receiver):
        await self.send_command(loop=loop, receiver=receiver, command='step')

    async def send_check_command(self, loop, receiver):
        await self.send_command(loop=loop, receiver=receiver, command='check')

    async def send_size_command(self, loop, receiver):
        await self.send_command(loop=loop, receiver=receiver, command='size')

    async def send_compute_input_command(self, loop, receiver):
        await self.send_command(loop=loop, receiver=receiver, command='compute_in')

    async def send_compute_output_command(self, loop, receiver):
        await self.send_command(loop=loop, receiver=receiver, command='compute_out')

    async def send_get_input_command(self, loop, receiver):
        await self.send_command(loop=loop, receiver=receiver, command='get_in')

    async def send_get_output_command(self, loop, receiver):
        await self.send_command(loop=loop, receiver=receiver, command='get_out')
