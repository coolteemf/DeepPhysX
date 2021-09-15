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

        recv_param_dict = {}
        # Receive parameters
        await self.listen_while_not_done(loop=loop, sender=self.sock, data_dict=recv_param_dict)

        # Use received parameters
        self.recv_parameters(recv_param_dict)

        # Create the environment
        self.create()
        self.init()

        # Send visualization
        visu_dict = self.send_visualization()
        for key in visu_dict:
            await self.send_command_read(loop=loop, receiver=self.sock)
            await self.send_labeled_data(data_to_send=visu_dict[key], label=key, loop=loop, receiver=self.sock)
        await self.send_command_done(loop=loop, receiver=self.sock)

        # Send parameters
        param_dict = self.send_parameters()
        for key in param_dict:
            await self.send_command_read(loop=loop, receiver=self.sock)
            # Send the parameter (label + data)
            await self.send_labeled_data(data_to_send=param_dict[key], label=key, loop=loop, receiver=self.sock,
                                         do_convert=key != 'addvedo')
            # Tell the client to stop receiving data
        await self.send_command_done(loop=loop, receiver=self.sock)

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
        if command not in self.command_dict.values():
            raise ValueError(f"Unknown command {command}")

        # 'exit': close the environment and the client
        if command == b'exit':
            self.close_client = True
        # 'step': trigger a step in the environment
        elif command == b'step':
            self.compute_essential_data = False
            await self.step()
        # 'cmpt': trigger a step in the environment and call for the sending of the training data
        elif command == b'cmpt':
            self.compute_essential_data = True
            await self.step()
        # 'test': check if the sample is exploitable
        elif command == b'test':
            check = b'1' if self.checkSample() else b'0'
            await self.send_data(data_to_send=check, loop=loop, receiver=server, do_convert=False)
        # 'pred': receive prediction and apply it
        elif command == b'pred':
            prediction = await self.receive_data(loop=loop, sender=server)
            self.applyPrediction(prediction.reshape(self.output_size))
        elif command == b'samp':
            sample_in = await self.receive_data(loop=loop, sender=server)
            sample_out = await self.receive_data(loop=loop, sender=server)
            self.setDatasetSample(sample_in, sample_out)

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

    async def send_training_data(self, network_input=None, network_output=None, loop=None, receiver=None):
        """

        :param loop: asyncio.get_event_loop() return
        :param receiver: TcpIpObject receiver
        :param network_input: data to send under the label \"input\"
        :param network_output: data to send under the label \"output\"
        :return:
        """
        loop = asyncio.get_event_loop() if loop is None else loop
        receiver = self.sock if receiver is None else receiver
        check = self.checkSample()
        await self.send_command_read()
        await self.send_labeled_data(data_to_send=b'1' if check else b'0', label="check", loop=loop, receiver=receiver,
                                     do_convert=False)
        if check:
            if network_input is not None:
                await self.send_command_read()
                await self.send_labeled_data(data_to_send=network_input, label="input", loop=loop, receiver=receiver)
            if network_output is not None:
                await self.send_command_read()
                await self.send_labeled_data(data_to_send=network_output, label="output", loop=loop, receiver=receiver)

    def sync_send_training_data(self, network_input=None, network_output=None, receiver=None):
        """

        :param receiver: TcpIpObject receiver
        :param network_input: data to send under the label \"input\"
        :param network_output: data to send under the label \"output\"
        :return:
        """
        receiver = self.sock if receiver is None else receiver
        check = self.checkSample()
        self.sync_send_command_read()
        self.sync_send_labeled_data(data_to_send=b'1' if check else b'0', label="check", receiver=receiver,
                                    do_convert=False)
        if check:
            if network_input is not None:
                self.sync_send_command_read()
                self.sync_send_labeled_data(data_to_send=network_input, label="input", receiver=receiver)
            if network_output is not None:
                self.sync_send_command_read()
                self.sync_send_labeled_data(data_to_send=network_output, label="output", receiver=receiver)

    def sync_send_prediction_request(self, network_input=None, receiver=None):
        """

        :param network_input: Data to send under the label 'input'
        :param receiver: TcpIpObject receiver
        :return:
        """
        receiver = self.sock if receiver is None else receiver
        if network_input is not None:
            self.sync_send_command_prediction()
            self.sync_send_labeled_data(data_to_send=network_input, label='input', receiver=receiver)
            label, pred = self.sync_receive_labeled_data()
            return pred

    def sync_send_visualization_data(self, visualization_data=None, receiver=None):
        """

        :param visualization_data:
        :param receiver:
        :return:
        """
        receiver = self.sock if receiver is None else receiver
        if visualization_data is not None:
            self.sync_send_command_visualization()
            for key in visualization_data:
                self.sync_send_command_read()
                self.sync_send_labeled_data(data_to_send=visualization_data[key], label=key, receiver=receiver)
            self.sync_send_command_done(receiver=self.sock)
