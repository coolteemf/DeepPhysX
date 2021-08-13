import asyncio
import numpy as np

from DeepPhysX_Core.AsyncSocket.TcpIpObject import TcpIpObject, BytesNumpyConverter
from DeepPhysX_Core.AsyncSocket.AbstractEnvironment import AbstractEnvironment


class TcpIpClient(TcpIpObject, AbstractEnvironment):

    def __init__(self, ip_address='localhost', port=10000, data_converter=BytesNumpyConverter, instance_id=1):
        """
        TcpIpClient is a TcpIpObject which communicate with a TcpIpServer and an AbstractEnvironment to compute data.

        :param str ip_address: IP address of the TcpIpObject
        :param int port: Port number of the TcpIpObject
        :param data_converter: BytesBaseConverter class to convert data to bytes (NumPy by default)
        :param int instance_id: ID of the instance
        """
        TcpIpObject.__init__(self, ip_address=ip_address, port=port, data_converter=data_converter)
        AbstractEnvironment.__init__(self, instance_id=instance_id)
        # Bind to client address
        self.sock.connect((ip_address, port))
        # Flag to trigger client's shutdown
        self.close_client = False

    def initialize(self):
        """
        Run __initialize method with asyncio.

        :return:
        """
        asyncio.run(self.__initialize())

    async def __initialize(self):
        """
        Receive parameters from the server to create environment, send parameters to the server in exchange.

        :return:
        """
        loop = asyncio.get_event_loop()

        # Receive parameters
        cmd = b''
        recv_param_dict = {}
        # Receive data while the server did not say to stop
        while cmd != b'done':
            # Receive and check server command
            cmd = await self.receive_data(loop, self.sock, is_bytes_data=True)
            if cmd not in [b'done', b'recv']:
                raise ValueError(f"Unknown command {cmd}, must be in {[b'done', b'recv']}")
            # Receive data if the server did not say to stop
            if cmd != b'done':
                label, param = await self.receive_labeled_data(loop, self.sock)
                recv_param_dict[label] = param
        # Use received parameters
        self.recv_parameters(recv_param_dict)

        # Send parameters
        param_dict = self.send_parameters()
        for key in param_dict:
            # Prepare the server to receive data
            await self.send_command_receive(loop=loop, receiver=self.sock)
            # Send the parameter (label + data)
            await self.send_labeled_data(data_to_send=param_dict[key], label=key, loop=loop, receiver=self.sock)
        # Tell the client to stop receiving data
        await self.send_command_done(loop=loop, receiver=self.sock)

        # Create the environment
        self.create()
        self.init()

        # Receive and check last init server command
        cmd = await self.receive_data(loop, self.sock, is_bytes_data=True)
        if cmd not in [b'done', b'size']:
            raise ValueError(f"Unknown command {cmd}, must be in {[b'done', b'size']}")
        # If server asked to send IO sizes
        if cmd == b'size':
            # Send input and output sizes
            await self.send_data(data_to_send=np.array(self.input_size, dtype=float), loop=loop,
                                 receiver=self.sock)
            await self.send_data(data_to_send=np.array(self.output_size, dtype=float), loop=loop,
                                 receiver=self.sock)

    def run(self):
        """
        Run __run method with asyncio.

        :return:
        """
        asyncio.run(self.__run())

    async def __run(self):
        """
        Trigger the main communication protocol with the server.

        :return:
        """
        loop = asyncio.get_event_loop()
        try:
            # Run the communication protocol with server while client is not asked to shutdown
            while not self.close_client:
                await self.__communicate(server=self.sock)
        except KeyboardInterrupt:
            print("KEYBOARD INTERRUPT: CLOSING PROCEDURE")
        finally:
            await self.__close()

    async def __communicate(self, server=None):
        """
        Communication protocol with a server. First receive a command from the client, then process the appropriate
        actions.

        :param server: TcpIpServer to communicate with
        :return:
        """
        loop = asyncio.get_event_loop()

        # Receive and check command
        command = await self.receive_data(loop, server, is_bytes_data=True)
        if command not in self.available_commands:
            raise ValueError(f"Unknown command {command}")

        # 'exit': close the environment and the client
        if command == b'exit':
            self.close_client = True

        # 'step': trigger a step in the environment
        elif command == b'step':
            self.step()

        # 'test': check if the sample is exploitable
        elif command == b'test':
            check = b'1' if self.checkSample() else b'0'
            await self.send_data(data_to_send=check, loop=loop, receiver=server, do_convert=False)

        # 'c_in': compute the environment input
        elif command == b'c_in':
            self.computeInput()

        # 'c_ou': compute the environment output
        elif command == b'c_ou':
            self.computeOutput()

        # 'g_in': send the environment input
        elif command == b'g_in':
            data_in = self.getInput()
            await self.send_data(data_to_send=data_in, loop=loop, receiver=server)

        # 'g_ou': send the environment output
        elif command == b'g_ou':
            data_out = self.getOutput()
            await self.send_data(data_to_send=data_out, loop=loop, receiver=server)

        # 'pred': receive prediction and apply it
        elif command == b'pred':
            prediction = await self.receive_data(loop=loop, sender=server)
            self.applyPrediction(prediction.reshape(self.output_size))

    async def __close(self):
        """
        Close the environment and shutdown the client.

        :return:
        """
        # Close environment
        self.close()

        # Shutdown client
        loop = asyncio.get_event_loop()
        # Confirm exit command to the server
        await self.send_command_exit(loop=loop, receiver=self.sock)
        # Close socket
        self.sock.close()
