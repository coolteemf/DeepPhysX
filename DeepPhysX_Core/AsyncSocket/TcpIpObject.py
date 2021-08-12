import socket

from DeepPhysX_Core.AsyncSocket.BytesNumpyConverter import BytesNumpyConverter


class TcpIpObject:

    def __init__(self, ip_address='localhost', port=10000, data_converter=BytesNumpyConverter):
        """
        TcpIpObject defines communication protocols to send and receive data and commands.

        :param str ip_address: IP address of the TcpIpObject
        :param int port: Port number of the TcpIpObject
        :param data_converter: BytesBaseConverter class to convert data to bytes (NumPy by default)
        """
        # Define socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Register IP and PORT
        self.ip_address = ip_address
        self.port = port
        # Create data converter
        self.data_converter = data_converter()
        # Available commands
        self.available_commands = [b'exit', b'step', b'test', b'size', b'done', b'recv',
                                   b'c_in', b'c_ou', b'g_in', b'g_ou']
        self.command_dict = {'exit': b'exit', 'step': b'step', 'check': b'test', 'size': b'size', 'done': b'done',
                             'recv': b'recv', 'compute_in': b'c_in', 'compute_out': b'c_ou', 'get_in': b'g_in',
                             'get_out': b'g_ou'}

    async def send_data(self, data_to_send, loop, receiver, do_convert=True):
        """
        Send data from a TcpIpObject to another.

        :param data_to_send: Data that will be sent on socket
        :param loop: asyncio.get_event_loop() return
        :param receiver: TcpIpObject receiver
        :param bool do_convert: Data will be converted in bytes by default. If the data is already in bytes, set to
               False
        :return:
        """
        # Cast data to bytes field
        data_as_bytes = self.data_converter.data_to_bytes(data_to_send) if do_convert else data_to_send
        # Size of tha data to send
        data_size = len(data_as_bytes)
        # Send the size of the next receive
        await loop.sock_sendall(sock=receiver, data=data_size.to_bytes(4, byteorder='big'))
        # Send the actual data
        await loop.sock_sendall(sock=receiver, data=data_as_bytes)

    async def receive_data(self, loop, sender, is_bytes_data=False):
        """
        Receive data from another TcpIpObject.

        :param loop: asyncio.get_event_loop() return
        :param sender: TcpIpObject sender
        :param is_bytes_data: Data will be converted from bytes by default. If the expected data is in bytes, set to
               True
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
            data_size_to_read -= len(data_received_as_bytes)
        # Return the data in the expected format
        return data_as_bytes if is_bytes_data else self.data_converter.bytes_to_data(data_as_bytes)

    async def send_labeled_data(self, data_to_send, label, loop, receiver, do_convert=True):
        """
        Send data with an associated label.

        :param data_to_send: Data that will be sent on socket
        :param str label: Associated label
        :param loop: asyncio.get_event_loop() return
        :param receiver: TcpIpObject receiver
        :param bool do_convert: Data will be converted in bytes by default. If the data is already in bytes, set to
               False
        :return:
        """
        await self.send_data(data_to_send=label, loop=loop, receiver=receiver, do_convert=False)
        await self.send_data(data_to_send=data_to_send, loop=loop, receiver=receiver, do_convert=do_convert)

    async def receive_labeled_data(self, loop, sender, is_bytes_data=False):
        """
        Receive data and an associated label.

        :param loop: asyncio.get_event_loop() return
        :param sender: TcpIpObject sender
        :param is_bytes_data: Data will be converted from bytes by default. If the expected data is in bytes, set to
               True
        :return:
        """
        label = await self.receive_data(loop=loop, sender=sender, is_bytes_data=True)
        data = await self.receive_data(loop=loop, sender=sender, is_bytes_data=is_bytes_data)
        return label, data

    async def send_command(self, loop, receiver, command=''):
        """
        Send a bytes command among the available commands.

        :param loop: asyncio.get_event_loop() return
        :param receiver: TcpIpObject receiver
        :param str command: Name of the command (see self.command_dict)
        :return:
        """
        # Check if the command exists
        try:
            cmd = self.command_dict[command]
        except KeyError:
            raise KeyError(f"\"{command}\" is not a valid command. Use {self.command_dict.keys()} instead.")
        # Send command as a byte data
        await self.send_data(data_to_send=cmd, loop=loop, receiver=receiver, do_convert=False)

    async def send_command_exit(self, loop, receiver):
        await self.send_command(loop=loop, receiver=receiver, command='exit')

    async def send_command_step(self, loop, receiver):
        await self.send_command(loop=loop, receiver=receiver, command='step')

    async def send_command_check(self, loop, receiver):
        await self.send_command(loop=loop, receiver=receiver, command='check')

    async def send_command_size(self, loop, receiver):
        await self.send_command(loop=loop, receiver=receiver, command='size')

    async def send_command_done(self, loop, receiver):
        await self.send_command(loop=loop, receiver=receiver, command='done')

    async def send_command_receive(self, loop, receiver):
        await self.send_command(loop=loop, receiver=receiver, command='recv')

    async def send_command_compute_input(self, loop, receiver):
        await self.send_command(loop=loop, receiver=receiver, command='compute_in')

    async def send_command_compute_output(self, loop, receiver):
        await self.send_command(loop=loop, receiver=receiver, command='compute_out')

    async def send_command_get_input(self, loop, receiver):
        await self.send_command(loop=loop, receiver=receiver, command='get_in')

    async def send_command_get_output(self, loop, receiver):
        await self.send_command(loop=loop, receiver=receiver, command='get_out')
