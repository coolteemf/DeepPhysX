import socket
import asyncio

from DeepPhysX_Core.AsyncSocket.BytesNumpyConverter import BytesNumpyConverter

import threading


def launchInThread(func):
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs)
        thread.start()
        return thread
    return wrapper


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
                             'received': b'recv', 'prediction': b'pred', 'compute': b'cmpt', 'read': b'read',
                             'sample': b'samp'}
        # Synchronous variables
        self.send_lock = threading.Lock()
        self.receive_lock = threading.Lock()

    # Asynchronous definition of the functions
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
        data_as_bytes = self.data_converter.data_to_bytes(data_to_send) if type(data_to_send) == self.data_converter.data_type() else data_to_send
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

    async def send_labeled_data(self, data_to_send, label, receiver, loop=None, do_convert=True):
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
        await self.send_data(data_to_send=bytes(label.lower(), "utf-8"), loop=loop, receiver=receiver, do_convert=False)
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
        data = await self.receive_data(loop=loop, sender=sender, is_bytes_data=True)
        if data in self.command_dict.values():
            label = (await self.receive_data(loop=loop, sender=sender, is_bytes_data=True)).decode("utf-8")
        else:
            label = data.decode("utf-8")
        if label in ["check", "addvedo"]:
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
        while await self.receive_data(loop=loop, sender=sender, is_bytes_data=True) != self.command_dict['done']:
            label, param = await self.receive_labeled_data(loop=loop, sender=sender)
            data_dict[label] = param

        # ############ This is debug mode of above. DO NOT ERASE
        # import time
        # while True:
        #     cmd = await self.receive_data(loop=loop, sender=sender, is_bytes_data=True)
        #     time.sleep(2)
        #     print(f"CMD ::: {cmd}")
        #     if cmd == b'done':
        #         break
        #     label, param = await self.receive_labeled_data(loop=loop, sender=sender)
        #     print(f"received {label=}")
        #     #print(f"value = {param}")
        #     data_dict[label] = param

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
    def sync_send_data(self, data_to_send, receiver=None, do_convert=True):
        """
        Send data from a TcpIpObject to another.

        :param data_to_send: Data that will be sent on socket
        :param receiver: TcpIpObject receiver
        :param bool do_convert: Data will be converted in bytes by default. If the data is already in bytes, set to
               False
        :return:
        """
        receiver = self.sock if receiver is None else receiver
        # Cast data to bytes field
        data_as_bytes = self.data_converter.data_to_bytes(data_to_send) if do_convert else data_to_send
        # Size of tha data to send
        data_size = len(data_as_bytes)
        # Send the size of the next receive
        #self.send_lock.acquire()
        receiver.sendall(data_size.to_bytes(4, byteorder='big'))
        #self.send_lock.release()
        # Send the actual data
        #self.send_lock.acquire()
        receiver.sendall(data_as_bytes)
        #self.send_lock.release()

    #@launchInThread
    def sync_receive_data(self, is_bytes_data=False):
        """
        Receive data from another TcpIpObject.

        :param is_bytes_data: Data will be converted from bytes by default. If the expected data is in bytes, set to
               True
        :return:
        """
        # Maximum read sizes array
        read_sizes = [4096, 2048, 1024, 512, 256]
        read_size_idx = 0
        self.sock.setblocking(True)
        # Always expect to receive the size of the data to read first
        #self.receive_lock.acquire()
        data_size_to_read = int.from_bytes(self.sock.recv(4), 'big')
        #self.receive_lock.release()
        data_as_bytes = b''
        # Proceed to read chunk by chunk
        while data_size_to_read > 0:
            # Select the good amount of bytes to read
            while read_size_idx < len(read_sizes) and data_size_to_read < read_sizes[read_size_idx]:
                read_size_idx += 1
            # If the amount of bytes to read is too small then read it all
            chunk_size_to_read = data_size_to_read if read_size_idx >= len(read_sizes) else read_sizes[read_size_idx]
            # Try to read at most "chunk_size_to_read" bytes from the socket
            #self.receive_lock.acquire()
            data_received_as_bytes = self.sock.recv(chunk_size_to_read)
            #self.receive_lock.release()
            # Accumulate the data
            data_as_bytes += data_received_as_bytes
            data_size_to_read -= len(data_received_as_bytes)
        # Return the data in the expected format
        return data_as_bytes if is_bytes_data else self.data_converter.bytes_to_data(data_as_bytes)

    # Functions below might not need the thread thingy
    #@launchInThread
    def sync_send_labeled_data(self, data_to_send, label, receiver=None, do_convert=True):
        """
        Send data with an associated label.

        :param data_to_send: Data that will be sent on socket
        :param str label: Associated label
        :param receiver: TcpIpObject receiver
        :param bool do_convert: Data will be converted in bytes by default. If the data is already in bytes, set to
               False
        :return:
        """
        self.sync_send_command_read()
        self.sync_send_data(data_to_send=bytes(label.lower(), "utf-8"), receiver=receiver, do_convert=False)
        self.sync_send_data(data_to_send=data_to_send, receiver=receiver, do_convert=do_convert)

    #@launchInThread
    def sync_receive_labeled_data(self, is_bytes_data=False):
        """
        Receive data and an associated label.

        :param sender: TcpIpObject sender
        :param is_bytes_data: Data will be converted from bytes by default. If the expected data is in bytes, set to
               True
        :return:
        """
        data = self.sync_receive_data(is_bytes_data=True)
        if data in self.command_dict.values():
            label = self.sync_receive_data(is_bytes_data=True).decode("utf-8")
        else:
            label = data.decode("utf-8")
        if label == "check":
            is_bytes_data = True
        data = self.sync_receive_data(is_bytes_data=is_bytes_data)
        return label, data

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
        self.sync_send_data(data_to_send=cmd, receiver=receiver, do_convert=False)

    def sync_listen_while_not_done(self, data_dict):
        while self.sync_receive_data(is_bytes_data=True) != self.command_dict['done']:
            label, param = self.sync_receive_labeled_data()
            data_dict[label] = param

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