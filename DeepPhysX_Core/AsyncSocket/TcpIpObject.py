from socket import socket, AF_INET, SOCK_STREAM, SOL_SOCKET, SO_REUSEADDR
from asyncio import get_event_loop
import threading

from DeepPhysX_Core.AsyncSocket.BytesConverter import BytesConverter


def launchInThread(func):
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs)
        thread.start()
        return thread
    return wrapper


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
        self.command_dict = {'exit': b'exit', 'step': b'step', 'done': b'done', 'finished': b'fini',
                             'prediction': b'pred', 'compute': b'cmpt', 'read': b'read', 'sample': b'samp',
                             'visualisation': b'visu'}
        self.action_on_command = {
            self.command_dict["exit"]: self.action_on_exit,
            self.command_dict["step"]: self.action_on_step,
            self.command_dict["done"]: self.action_on_done,
            self.command_dict["finished"]: self.action_on_finished,
            self.command_dict["prediction"]: self.action_on_prediction,
            self.command_dict["compute"]: self.action_on_compute,
            self.command_dict["read"]: self.action_on_read,
            self.command_dict["sample"]: self.action_on_sample,
            self.command_dict["visualisation"]: self.action_on_visualisation,
        }
        # Synchronous variables
        # self.send_lock = Lock()
        # self.receive_lock = Lock()

    ##########################################################################################
    ##########################################################################################
    #                      LOW level of send & receive data on network                       #
    ##########################################################################################
    ##########################################################################################

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
        if await loop.sock_sendall(sock=receiver, data=data_as_bytes) is not None:
            ValueError("Could not send all of the data for an unknown reason")

    def sync_send_data(self, data_to_send, receiver=None):
        """
        Send data from a TcpIpObject to another.
        Synchronous version of 'TcpIpObject.send_data'.

        :param data_to_send: Data that will be sent on socket
        :param receiver: TcpIpObject receiver
        :return:
        """

        receiver = self.sock if receiver is None else receiver
        # Cast data to bytes fields
        data_as_bytes = self.data_converter.data_to_bytes(data_to_send)
        # Send the whole message
        receiver.sendall(data_as_bytes)

    async def receive_data(self, loop, sender):
        """
        Receive data from another TcpIpObject.

        :param loop: asyncio.get_event_loop() return
        :param sender: TcpIpObject sender
        :return: Converted data
        """

        # Receive the number of fields to receive
        nb_bytes_fields_b = await loop.sock_recv(sender, self.data_converter.int_size)
        nb_bytes_fields = self.data_converter.size_from_bytes(nb_bytes_fields_b)
        # Receive the sizes in bytes of all the relevant fields
        sizes_b = [await loop.sock_recv(sender, self.data_converter.int_size) for _ in range(nb_bytes_fields)]
        sizes = [self.data_converter.size_from_bytes(size_b) for size_b in sizes_b]
        # Receive each byte field
        bytes_fields = [await self.read_data(loop, sender, size) for size in sizes]
        # Return the data in the expected format
        return self.data_converter.bytes_to_data(bytes_fields)

    def sync_receive_data(self):
        """
        Receive data from another TcpIpObject.
        Synchronous version of 'TcpIpObject.receive_data'.

        :return:
        """
        self.sock.setblocking(True)
        # Receive the number of fields to receive
        nb_bytes_fields_b = self.sock.recv(self.data_converter.int_size)
        nb_bytes_fields = self.data_converter.size_from_bytes(nb_bytes_fields_b)
        # Receive the sizes in bytes of all the relevant fields
        sizes_b = [self.sock.recv(self.data_converter.int_size) for _ in range(nb_bytes_fields)]
        sizes = [self.data_converter.size_from_bytes(size_b) for size_b in sizes_b]
        # Receive each byte field
        bytes_fields = [self.sync_read_data(size) for size in sizes]
        # Return the data in the expected format
        return self.data_converter.bytes_to_data(bytes_fields)

    async def read_data(self, loop, sender, read_size):
        """
        Read the data on the socket with value of buffer size as relatively small powers of 2.

        :param loop: asyncio.get_event_loop() return
        :param sender: TcpIpObject sender
        :param int read_size: Amount of data to read on the socket
        :return: bytes - Bytes field with 'read_size' length
        """

        # Maximum read size
        read_sizes = [4096]
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

    def sync_read_data(self, read_size):
        """
        Read the data on the socket with value of buffer size as relatively small powers of 2.
        Synchronous version of 'TcpIpObject.read_data'.

        :param int read_size: Amount of data to read on the socket
        :return: bytes - Bytes field with 'read_size' length
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
            # Accumulate the data
            bytes_field += data_received_as_bytes
            read_size -= len(data_received_as_bytes)

        return bytes_field

    ##########################################################################################
    ##########################################################################################
    #                            Send & receive abstract named data                          #
    ##########################################################################################
    ##########################################################################################

    async def send_labeled_data(self, data_to_send, label, loop=None, receiver=None, send_read_command=True):
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
        # Send a 'read' command before data if specified
        if send_read_command:
            await self.send_command_read(loop=loop, receiver=receiver)
        # Send label
        await self.send_data(data_to_send=label, loop=loop, receiver=receiver)
        # Send data
        await self.send_data(data_to_send=data_to_send, loop=loop, receiver=receiver)

    def sync_send_labeled_data(self, data_to_send, label, receiver=None, send_read_command=True):
        """
        Send data with an associated label.
        Synchronous version of 'TcpIpObject.send_labeled_data'.

        :param data_to_send: Data that will be sent on socket
        :param str label: Associated label
        :param receiver: TcpIpObject receiver
        :param bool send_read_command: If True, the command 'read' is sent before sending data
        :return:
        """

        receiver = self.sock if receiver is None else receiver
        # Send a 'read' command before data if specified
        if send_read_command:
            self.sync_send_command_read(receiver=receiver)
        # Send label
        self.sync_send_data(data_to_send=label, receiver=receiver)
        # Send data
        self.sync_send_data(data_to_send=data_to_send, receiver=receiver)

    async def receive_labeled_data(self, loop, sender):
        """
        Receive data and an associated label.

        :param loop: asyncio.get_event_loop() return
        :param sender: TcpIpObject sender
        :return: str, _ - Label, Data
        """

        # Listen to sender
        recv = await self.receive_data(loop=loop, sender=sender)
        # 'recv' can be either a 'read' command either the label
        if recv in self.command_dict.values():
            label = await self.receive_data(loop=loop, sender=sender)
        else:
            label = recv
        # Receive data
        data = await self.receive_data(loop=loop, sender=sender)
        return label, data

    def sync_receive_labeled_data(self):
        """
        Receive data and an associated label.
        Synchronous version of 'TcpIpObject.receive_labeled_data'.

        :return: str, _ - Label, Data
        """

        # Listen to sender
        recv = self.sync_receive_data()
        # 'recv' can be either a 'read' command either the label
        if recv in self.command_dict.values():
            label = self.sync_receive_data()
        else:
            label = recv
        # Receive data
        data = self.sync_receive_data()
        return label, data

    async def send_dict(self, name, dict_to_send, loop=None, receiver=None):
        """
        Send a whole dictionary field by field as labeled data.

        :param str name: Name of the dictionary
        :param dict dict_to_send: Dictionary to send
        :param loop: asyncio.get_event_loop() return
        :param receiver: TcpIpObject receiver
        :return:
        """

        loop = get_event_loop() if loop is None else loop
        receiver = self.sock if receiver is None else receiver

        # If dict is empty, the sending is finished
        if dict_to_send is None or dict_to_send == {}:
            await self.send_command_finished(loop=loop, receiver=receiver)
            return

        # Sends to make the listener start the receive_dict routine
        await self.send_command_read(loop=loop, receiver=receiver)
        await self.send_labeled_data(data_to_send=name, label="::dict::", loop=loop, receiver=receiver)

        # Treat the dictionary field by field
        for key in dict_to_send:
            # If data is another dict, send as an unnamed dictionary
            if type(dict_to_send[key]) == dict:
                # Send key
                await self.send_labeled_data(data_to_send=key, label="dict_id", loop=loop, receiver=receiver)
                # Send data
                await self.__send_unnamed_dict(dict_to_send=dict_to_send[key], loop=loop, receiver=receiver)
            # If data is not a dict, send as labeled data
            else:
                # Send key and data
                await self.send_labeled_data(data_to_send=dict_to_send[key], label=key, loop=loop, receiver=receiver)

        # The sending is finished
        await self.send_command_finished(loop=loop, receiver=receiver)
        await self.send_command_finished(loop=loop, receiver=receiver)

    def sync_send_dict(self, name, dict_to_send, receiver=None):
        """
        Send a whole dictionary field by field as labeled data.
        Synchronous version of 'TcpIpObject.receive_labeled_data'.

        :param str name: Name of the dictionary
        :param dict dict_to_send: Dictionary to send
        :param receiver: TcpIpObject receiver
        :return:
        """

        receiver = self.sock if receiver is None else receiver

        # If dict is empty, the sending is finished
        if dict_to_send is None or dict_to_send == {}:
            self.sync_send_command_finished(receiver=receiver)
            return

        # Sends to make the listener start the receive_dict routine
        self.sync_send_command_read()
        self.sync_send_labeled_data(data_to_send=name, label="::dict::", receiver=receiver)

        # Treat the dictionary field by field
        for key in dict_to_send:
            # If data is another dict, send as an unnamed dictionary
            if type(dict_to_send[key]) == dict:
                # Send key
                self.sync_send_labeled_data(data_to_send=key, label="dict_id", receiver=receiver)
                # Send data
                self.__sync_send_unnamed_dict(dict_to_send=dict_to_send[key], receiver=receiver)
            # If data is not a dict, send as labeled data
            else:
                # Send key and data
                self.sync_send_labeled_data(data_to_send=dict_to_send[key], label=key, receiver=receiver)

        # The sending is finished
        self.sync_send_command_finished(receiver=receiver)
        self.sync_send_command_finished(receiver=receiver)

    async def __send_unnamed_dict(self, dict_to_send, loop=None, receiver=None, blank=''):
        """
        Send a whole dictionary field by field as labeled data. Dictionary will be unnamed.

        :param dict dict_to_send: Dictionary to send
        :param loop: asyncio.get_event_loop() return
        :param receiver: TcpIpObject receiver
        :param str blank: Leave empty spaces to dissociate recursive dictionaries
        :return:
        """

        blank += '  '
        loop = get_event_loop() if loop is None else loop
        receiver = self.sock if receiver is None else receiver

        # Treat the dictionary field by field
        for key in dict_to_send:
            # If data is another dict, send as an unnamed dictionary
            if type(dict_to_send[key]) == dict:
                # Send key
                await self.send_labeled_data(data_to_send=key, label="dict_id", loop=loop, receiver=receiver)
                # Send data
                await self.__send_unnamed_dict(dict_to_send=dict_to_send[key], loop=loop, receiver=receiver, blank=blank)
            # If data is not a dict, send as labeled data
            else:
                # Send key and data
                await self.send_labeled_data(data_to_send=dict_to_send[key], label=key, loop=loop, receiver=receiver)

        # The sending is finished
        await self.send_command_finished(loop=loop, receiver=receiver)

    def __sync_send_unnamed_dict(self, dict_to_send, receiver=None, blank=''):
        """
        Send a whole dictionary field by field as labeled data. Dictionary will be unnamed.
        Synchronous version of 'TcpIpObject.receive_labeled_data'.

        :param dict dict_to_send: Dictionary to send
        :param receiver: TcpIpObject receiver
        :param str blank: Leave empty spaces to dissociate recursive dictionaries
        :return:
        """

        blank += '  '
        receiver = self.sock if receiver is None else receiver

        # Treat the dictionary field by field
        for key in dict_to_send:
            # If data is another dict, send as an unnamed dictionary
            if type(dict_to_send[key]) == dict:
                # Send key
                self.sync_send_labeled_data(data_to_send=key, label="dict_id", receiver=receiver, send_read_command=True)
                # Send data
                self.__sync_send_unnamed_dict(dict_to_send=dict_to_send[key], receiver=receiver, blank=blank)
            # If data is not a dict, send as labeled data
            else:
                # Send key and data
                self.sync_send_labeled_data(data_to_send=dict_to_send[key], label=key, receiver=receiver, send_read_command=True)

        # The sending is finished
        self.sync_send_command_finished(receiver=receiver)

    async def receive_dict(self, recv_to, loop=None, sender=None):
        """
        Receive a whole dictionary field by field as labeled data.

        :param dict recv_to: Dictionary to fill with received fields
        :param loop: asyncio.get_event_loop() return
        :param sender: TcpIpObject sender
        :return:
        """

        loop = get_event_loop() if loop is None else loop
        sender = self.sock if sender is None else sender

        # Receive data while command 'finished' is not received
        while (cmd := await self.receive_data(loop=loop, sender=sender)) != self.command_dict['finished']:
            # Receive field as a labeled data
            label, param = await self.receive_labeled_data(loop=loop, sender=sender)
            # If label refers to dict keyword, receive an unnamed dict
            if label in ["::dict::", "dict_id"]:
                recv_to[param] = {}
                await self.receive_dict(recv_to=recv_to[param], loop=loop, sender=sender)
            # Otherwise, set the dict field directly
            else:
                recv_to[label] = param

    def sync_receive_dict(self, recv_to, sender=None):
        """
        Receive a whole dictionary field by field as labeled data.
        Synchronous version of 'TcpIpObject.receive_labeled_data'.

        :param dict recv_to: Dictionary to fill with received fields
        :param sender: TcpIpObject sender
        :return:
        """

        sender = self.sock if sender is None else sender

        # Receive data while command 'finished' is not received
        while self.sync_receive_data() != self.command_dict['finished']:
            # Receive field as a labeled data
            label, param = self.sync_receive_labeled_data()
            # If label refers to dict keyword, receive an unnamed dict
            if label in ["::dict::", "dict_id"]:
                recv_to[param] = {}
                self.sync_receive_dict(recv_to=recv_to[param], sender=sender)
            # Otherwise, set the dict field directly
            else:
                recv_to[label] = param

    ##########################################################################################
    ##########################################################################################
    #                                 Command related sends                                  #
    ##########################################################################################
    ##########################################################################################

    async def send_command(self, loop, receiver, command=''):
        """
        Send a bytes command among the available commands.

        :param loop: asyncio.get_event_loop() return
        :param receiver: TcpIpObject receiver
        :param str command: Name of the command, must be in 'self.command_dict'
        :return:
        """

        # Check if the command exists
        try:
            cmd = self.command_dict[command]
        except KeyError:
            raise KeyError(f"\"{command}\" is not a valid command. Use {self.command_dict.keys()} instead.")
        # Send command as a byte data
        await self.send_data(data_to_send=cmd, loop=loop, receiver=receiver)

    def sync_send_command(self, receiver, command=''):
        """
        Send a bytes command among the available commands.
        Synchronous version of 'TcpIpObject.send_command'.

        :param receiver: TcpIpObject receiver
        :param str command: Name of the command, must be in 'self.command_dict'
        :return:
        """

        # Check if the command exists
        try:
            cmd = self.command_dict[command]
        except KeyError:
            raise KeyError(f"\"{command}\" is not a valid command. Use {self.command_dict.keys()} instead.")
        # Send command as a byte data
        self.sync_send_data(data_to_send=cmd, receiver=receiver)

    async def send_command_compute(self, loop=None, receiver=None):
        await self.send_command(loop=loop, receiver=receiver, command='compute')

    def sync_send_command_compute(self, receiver=None):
        self.sync_send_command(receiver=receiver, command='compute')

    async def send_command_done(self, loop=None, receiver=None):
        await self.send_command(loop=loop, receiver=receiver, command='done')

    def sync_send_command_done(self, receiver=None):
        self.sync_send_command(receiver=receiver, command='done')

    async def send_command_exit(self, loop=None, receiver=None):
        await self.send_command(loop=loop, receiver=receiver, command='exit')

    def sync_send_command_exit(self, receiver=None):
        self.sync_send_command(receiver=receiver, command='exit')

    async def send_command_finished(self, loop=None, receiver=None):
        await self.send_command(loop=loop, receiver=receiver, command='finished')

    def sync_send_command_finished(self, receiver=None):
        self.sync_send_command(receiver=receiver, command='finished')

    async def send_command_prediction(self, loop=None, receiver=None):
        await self.send_command(loop=loop, receiver=receiver, command='prediction')

    def sync_send_command_prediction(self, receiver=None):
        self.sync_send_command(receiver=receiver, command='prediction')

    async def send_command_read(self, loop=None, receiver=None):
        await self.send_command(loop=loop, receiver=receiver, command='read')

    def sync_send_command_read(self, receiver=None):
        self.sync_send_command(receiver=receiver, command='read')

    async def send_command_sample(self, loop=None, receiver=None):
        await self.send_command(loop=loop, receiver=receiver, command='sample')

    def sync_send_command_sample(self, receiver=None):
        self.sync_send_command(receiver=receiver, command='sample')

    async def send_command_step(self, loop=None, receiver=None):
        await self.send_command(loop=loop, receiver=receiver, command='step')

    def sync_send_command_step(self, receiver=None):
        self.sync_send_command(receiver=receiver, command='step')

    async def send_command_visualisation(self, loop=None, receiver=None):
        await self.send_command(loop=loop, receiver=receiver, command='visualisation')

    def sync_send_command_visualisation(self, receiver=None):
        self.sync_send_command(receiver=receiver, command='visualisation')

    ##########################################################################################
    ##########################################################################################
    #                            Actions to perform on commands                              #
    ##########################################################################################
    ##########################################################################################

    async def listen_while_not_done(self, loop, sender, data_dict, client_id=None):
        """
        Compute actions until 'done' command is received.

        :param loop: asyncio.get_event_loop() return
        :param sender: TcpIpObject sender
        :param dict data_dict: Dictionary to collect data
        :param client_id: ID of a Client
        :return:
        """

        # Compute actions until 'done' command is received
        while (cmd := await self.receive_data(loop=loop, sender=sender)) != self.command_dict['done']:
            # Compute the associated action
            if cmd in self.command_dict.values():
                await self.action_on_command[cmd](data=data_dict, client_id=client_id, sender=sender, loop=loop)
        # Return collected data
        return data_dict

    async def action_on_compute(self, data, client_id, sender, loop):
        pass

    async def action_on_done(self, data, client_id, sender, loop):
        pass

    async def action_on_exit(self, data, client_id, sender, loop):
        pass

    async def action_on_finished(self, data, client_id, sender, loop):
        pass

    async def action_on_prediction(self, data, client_id, sender, loop):
        pass

    async def action_on_read(self, data, client_id, sender, loop):
        """
        Define actions to take on a 'read' command.

        :param dict data: Dict storing data
        :param int client_id: ID of the TcpIpClient
        :param loop: asyncio.get_event_loop() return
        :param sender: TcpIpObject sender
        :return:
        """
        # Receive labeled data
        label, param = await self.receive_labeled_data(loop=loop, sender=sender)
        # If data to receive appears to be a dict, receive dict
        if param == "::dict::":
            data[client_id][label] = {}
            await self.receive_dict(recv_to=data[client_id][label], sender=sender, loop=loop)
        # Otherwise add labeled data to data dict
        else:
            data[client_id][label] = param

    async def action_on_sample(self, data, client_id, sender, loop):
        pass

    async def action_on_step(self, data, client_id, sender, loop):
        pass

    async def action_on_visualisation(self, data, client_id, sender, loop):
        pass
