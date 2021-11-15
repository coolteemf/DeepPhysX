from socket import socket, AF_INET, SOCK_STREAM, SOL_SOCKET, SO_REUSEADDR
from asyncio import get_event_loop
from threading import Lock

from DeepPhysX_Core.AsyncSocket.BytesConverter import BytesConverter

# import threading
#
#
# def launchInThread(func):
#     def wrapper(*args, **kwargs):
#         thread = threading.Thread(target=func, args=args, kwargs=kwargs)
#         thread.start()
#         return thread
#     return wrapper


class TcpIpObject:

    def __init__(self,
                 ip_address='localhost',
                 port=10000):
        """
        TcpIpObject defines communication protocols to send and receive data and commands.

        :param str ip_address: IP address of the TcpIpObject
        :param int port: Port number of the TcpIpObject
        """

        self.name = self.__class__.__name__

        # Define socket
        self.sock = socket(AF_INET, SOCK_STREAM)
        self.sock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
        # Register IP and PORT
        self.ip_address = ip_address
        self.port = port
        # Create data converter
        self.data_converter = BytesConverter()
        # Available commands
        self.command_dict = {'exit': b'exit', 'step': b'step', 'check': b'test', 'size': b'size', 'done': b'done',
                             'received': b'recv', 'prediction': b'pred', 'compute': b'cmpt', 'read': b'read',
                             'sample': b'samp', 'visualization': b'visu', 'unsupervised': b'uspv'}
        # Synchronous variables
        # self.send_lock = Lock()
        # self.receive_lock = Lock()

        self.times = []

    # Asynchronous definition of the functions
    async def send_data(self, data_to_send, loop=None, receiver=None):
        """
        Send data from a TcpIpObject to another.

        :param data_to_send: Data that will be sent on socket
        :param loop: asyncio.get_event_loop() return
        :param receiver: TcpIpObject receiver
        :return:
        """
        loop = get_event_loop() if loop is None else loop
        receiver = self.sock if receiver is None else receiver
        # Cast data to bytes fields
        data_as_bytes = self.data_converter.data_to_bytes(data_to_send)
        # Send the whole message
        await loop.sock_sendall(sock=receiver, data=data_as_bytes)

    async def receive_data(self, loop, sender):
        """
        Receive data from another TcpIpObject.

        :param loop: asyncio.get_event_loop() return
        :param sender: TcpIpObject sender
        :return: Converted data
        """
        # Receive the number of fields to receive
        nb_bytes_fields = self.data_converter.size_from_bytes(await loop.sock_recv(sender, 1))
        sizes = [self.data_converter.size_from_bytes(await loop.sock_recv(sender, 4)) for _ in range(nb_bytes_fields)]
        # bytes_fields = [await loop.sock_recv(sender, size) for size in sizes]
        bytes_fields = [await self.read_data(loop, sender, size) for size in sizes]
        # Return the data in the expected format
        return self.data_converter.bytes_to_data(bytes_fields)

    async def read_data(self, loop, sender, read_size):
        """
        Read the data on the socket with value of buffer size as relatively small powers of 2.

        :param loop: asyncio.get_event_loop() return
        :param sender: TcpIpObject sender
        :param read_size: Amount of data to read on the socket
        :return: Bytes field of read_size length
        """
        # Maximum read sizes array
        read_sizes = [8192, 4096, 2048, 1024, 512, 256]
        bytes_field = b''
        while read_size > 0:
            # Select the good amount of bytes to read
            read_size_idx = 0
            while read_size_idx < len(read_sizes) and read_size < read_sizes[read_size_idx]:
                read_size_idx += 1
            # If the amount of bytes to read is too small then read it all
            chunk_size_to_read = read_size if read_size_idx >= len(read_sizes) else read_sizes[read_size_idx]
            # Try to read at most chunk_size_to_read bytes from the socket
            data_received_as_bytes = await loop.sock_recv(sender, chunk_size_to_read)
            # Todo: add security with <<await asyncio.wait_for(loop.sock_recv(sender, chunk_size_to_read), timeout=1.)>>
            # Accumulate the data
            bytes_field += data_received_as_bytes
            read_size -= len(data_received_as_bytes)
        return bytes_field

    async def send_labeled_data(self, data_to_send, label, receiver=None, loop=None, send_read_command=True):
        """
        Send data with an associated label.

        :param data_to_send: Data that will be sent on socket
        :param str label: Associated label
        :param loop: asyncio.get_event_loop() return
        :param receiver: TcpIpObject receiver
        :param bool send_read_command: If True, the command 'read' is sent before sending data
        :return:
        """
        loop = get_event_loop() if loop is None else loop
        receiver = self.sock if receiver is None else receiver
        if send_read_command:
            await self.send_command_read(loop=loop, receiver=receiver)
        await self.send_data(data_to_send=label, loop=loop, receiver=receiver)
        await self.send_data(data_to_send=data_to_send, loop=loop, receiver=receiver)

    async def receive_labeled_data(self, loop, sender):
        """
        Receive data and an associated label.

        :param loop: asyncio.get_event_loop() return
        :param sender: TcpIpObject sender
        :return:
        """
        data = await self.receive_data(loop=loop, sender=sender)
        if data in self.command_dict.values():
            label = await self.receive_data(loop=loop, sender=sender)
        else:
            label = data
        data = await self.receive_data(loop=loop, sender=sender)
        return label, data

    async def send_dict_data(self, dict_data, receiver=None, loop=None):
        """
        Send a dictionary item by item.

        :param dict_data: Dict of data that will be sent on socket.
        :param receiver: TcpIpObject receiver.
        :param loop: asyncio.get_event_loop() return
        :return:
        """
        for field in dict_data.keys():
            await self.send_labeled_data(data_to_send=dict_data[field], label=field, receiver=receiver, loop=loop)
        await self.send_command_done(loop=loop, receiver=receiver)

    async def receive_dict_data(self, loop, sender):
        """
        Receive a dictionary item by item.

        :param loop: asyncio.get_event_loop() return
        :param sender: TcpIpObject sender
        :return:
        """
        dict_data = {}
        while await self.receive_data(loop=loop, sender=sender) != self.command_dict['done']:
            label, data = await self.receive_labeled_data(loop=loop, sender=sender)
            dict_data[label] = data
        return dict_data

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
        await self.send_data(data_to_send=cmd, loop=loop, receiver=receiver)

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
        await self.send_command(loop=loop, receiver=receiver, command='received')

    async def send_command_prediction(self, loop=None, receiver=None):
        await self.send_command(loop=loop, receiver=receiver, command='prediction')

    async def send_command_compute(self, loop=None, receiver=None):
        await self.send_command(loop=loop, receiver=receiver, command='compute')

    async def send_command_read(self, loop=None, receiver=None):
        await self.send_command(loop=loop, receiver=receiver, command='read')

    async def send_command_sample(self, loop=None, receiver=None):
        await self.send_command(loop=loop, receiver=receiver, command='sample')

    # Synchronous definition of the functions
    #@launchInThread
    def sync_send_data(self, data_to_send, receiver=None):
        """
        Send data from a TcpIpObject to another.

        :param data_to_send: Data that will be sent on socket
        :param receiver: TcpIpObject receiver
        :return:
        """
        receiver = self.sock if receiver is None else receiver
        # Cast data to bytes fields
        data_as_bytes = self.data_converter.data_to_bytes(data_to_send)
        # Send the whole message
        receiver.sendall(data_as_bytes)

    #@launchInThread
    def sync_receive_data(self):
        """
        Receive data from another TcpIpObject.

        :return:
        """
        self.sock.setblocking(True)
        # Receive the number of fields to receive
        nb_bytes_fields = self.data_converter.size_from_bytes(self.sock.recv(1))
        # Receive the sizes in bytes of all the relevant fields
        sizes = [self.data_converter.size_from_bytes(self.sock.recv(4)) for _ in range(nb_bytes_fields)]
        # Receive each bytes field
        # bytes_fields = [self.sock.recv(size) for size in sizes]
        bytes_fields = [self.sync_read_data(size) for size in sizes]
        # Return the data in the expected format
        return self.data_converter.bytes_to_data(bytes_fields)

    def sync_read_data(self, read_size):
        """
        Read the data on the socket with value of buffer size as relatively small powers of 2.

        :param read_size: Amount of data to read on the socket
        :return: Bytes field of read_size length
        """
        # Maximum read sizes array
        read_sizes = [8192, 4096, 2048, 1024, 512, 256]
        bytes_field = b''
        while read_size > 0:
            # Select the good amount of bytes to read
            read_size_idx = 0
            while read_size_idx < len(read_sizes) and read_size < read_sizes[read_size_idx]:
                read_size_idx += 1
            # If the amount of bytes to read is too small then read it all
            chunk_size_to_read = read_size if read_size_idx >= len(read_sizes) else read_sizes[read_size_idx]
            # Try to read at most chunk_size_to_read bytes from the socket
            data_received_as_bytes = self.sock.recv(chunk_size_to_read)
            # Todo: add security with <<await asyncio.wait_for(loop.sock_recv(sender, chunk_size_to_read), timeout=1.)>>
            # Accumulate the data
            bytes_field += data_received_as_bytes
            read_size -= len(data_received_as_bytes)
        return bytes_field

    # Functions below might not need the thread thingy
    #@launchInThread
    def sync_send_labeled_data(self, data_to_send, label, receiver=None, send_read_command=True):
        """
        Send data with an associated label.

        :param data_to_send: Data that will be sent on socket
        :param str label: Associated label
        :param receiver: TcpIpObject receiver
        :return:
        """
        if send_read_command:
            self.sync_send_command_read()
        self.sync_send_data(data_to_send=label, receiver=receiver)
        self.sync_send_data(data_to_send=data_to_send, receiver=receiver)

    #@launchInThread
    def sync_receive_labeled_data(self):
        """
        Receive data and an associated label.

        :return:
        """
        data = self.sync_receive_data()
        if data in self.command_dict.values():
            label = self.sync_receive_data()
        else:
            label = data
        data = self.sync_receive_data()
        return label, data

    def sync_send_dict_data(self, dict_data, receiver=None):
        """
        Send a dictionary item by item.

        :param dict_data:
        :param receiver:
        :return:
        """
        for field in dict_data.keys():
            self.sync_send_labeled_data(data_to_send=dict_data[field], label=field, receiver=receiver,
                                        send_read_command=True)
        self.sync_send_command_done()

    def sync_receive_dict_data(self):
        """
        Receive a dictionary item by item.

        :return:
        """
        dict_data = {}
        while self.sync_receive_data() != self.command_dict['done']:
            label, data = self.sync_receive_labeled_data()
            dict_data[label] = data
        return dict_data

    #@launchInThread
    def sync_send_command(self, receiver, command=''):
        """
        Send a bytes command among the available commands.

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
        self.sync_send_data(data_to_send=cmd, receiver=receiver)

    def sync_send_command_exit(self, receiver=None):
        self.sync_send_command(receiver=receiver, command='exit')

    def sync_send_command_step(self, receiver=None):
        self.sync_send_command(receiver=receiver, command='step')

    def sync_send_command_check(self, receiver=None):
        self.sync_send_command(receiver=receiver, command='check')

    def sync_send_command_size(self, receiver=None):
        self.sync_send_command(receiver=receiver, command='size')

    def sync_send_command_done(self, receiver=None):
        self.sync_send_command(receiver=receiver, command='done')

    def sync_send_command_received(self, receiver=None):
        self.sync_send_command(receiver=receiver, command='received')

    def sync_send_command_prediction(self, receiver=None):
        self.sync_send_command(receiver=receiver, command='prediction')

    def sync_send_command_compute(self, receiver=None):
        self.sync_send_command(receiver=receiver, command='compute')

    def sync_send_command_read(self, receiver=None):
        self.sync_send_command(receiver=receiver, command='read')

    def sync_send_command_visualization(self, receiver=None):
        self.sync_send_command(receiver=receiver, command='visualization')
