import socket
import asyncio

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
        self.command_dict = {'exit': b'exit', 'step': b'step', 'check': b'test', 'size': b'size', 'done': b'done',
                             'recv': b'recv', 'prediction': b'pred', 'compute': b'cmpt'}

    async def send_data(self, data_to_send, loop=None, receiver=None, do_convert=True):
        """
        Send data from a TcpIpObject to another.

        :param data_to_send: Data that will be sent on socket
        :param loop: asyncio.get_event_loop() return
        :param receiver: TcpIpObject receiver
        :param bool do_convert: Data will be converted in bytes by default. If the data is already in bytes, set to
               False
        :return:
        """
        loop = asyncio.get_event_loop() if loop is None else loop
        receiver = self.sock if receiver is None else receiver
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
        await self.send_data(data_to_send=bytes(label, "utf-8"), loop=loop, receiver=receiver, do_convert=False)
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
        label = (await self.receive_data(loop=loop, sender=sender, is_bytes_data=True)).decode("utf-8")
        if label == "check":
            is_bytes_data = True
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

    async def listen_while_not_done(self, loop, sender, data_dict):
        while await self.receive_data(loop=loop, sender=sender, is_bytes_data=True) != b'done':
            label, param = await self.receive_labeled_data(loop=loop, sender=sender)
            data_dict[label] = param

    async def send_command_exit(self, loop=None, receiver=None):
        await self.send_command(loop=loop, receiver=receiver, command='exit')

    async def send_command_step(self, loop=None, receiver=None):
        await self.send_command(loop=loop, receiver=receiver, command='step')

    async def send_command_check(self, loop=None, receiver=None):
        await self.send_command(loop=loop, receiver=receiver, command='check')

    async def send_command_size(self, loop=None, receiver=None):
        await self.send_command(loop=loop, receiver=receiver, command='size')

    async def send_command_done(self, loop=None, receiver=None):
        await self.send_command(loop=loop, receiver=receiver, command='done')

    async def send_command_received(self, loop=None, receiver=None):
        await self.send_command(loop=loop, receiver=receiver, command='recv')

    async def send_command_prediction(self, loop=None, receiver=None):
        await self.send_command(loop=loop, receiver=receiver, command='prediction')

    async def send_command_get_learning_data(self, loop=None, receiver=None):
        await self.send_command(loop=loop, receiver=receiver, command='compute')

    async def send_command_dummy(self, loop=None, receiver=None):
        await self.send_data(b'0', loop, receiver, do_convert=False)
