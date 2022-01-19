import socket
from asyncio import run, get_event_loop
from typing import Any, Optional, Dict

import numpy

from DeepPhysX_Core.AsyncSocket.TcpIpObject import TcpIpObject
from DeepPhysX_Core.AsyncSocket.AbstractEnvironment import AbstractEnvironment


class TcpIpClient(TcpIpObject, AbstractEnvironment):

    def __init__(self,
                 ip_address: str = 'localhost',
                 port: int = 10000,
                 instance_id: int = 1,
                 number_of_instances: int = 1,
                 as_tcp_ip_client: bool = True):
        """
        TcpIpClient is both a TcpIpObject which communicate with a TcpIpServer and an AbstractEnvironment to compute
        simulated data.

        :param str ip_address: IP address of the TcpIpObject
        :param int port: Port number of the TcpIpObject
        :param int instance_id: ID of the instance
        :param int number_of_instances: Number of simultaneously launched instances
        :param as_tcpip_client: Environment is a TcpIpObject if True, is owned by an EnvironmentManager if False
        """

        AbstractEnvironment.__init__(self,
                                     instance_id=instance_id,
                                     number_of_instances=number_of_instances,
                                     as_tcp_ip_client=as_tcp_ip_client)

        # Bind to client address
        if self.as_tcp_ip_client:
            TcpIpObject.__init__(self,
                                 ip_address=ip_address,
                                 port=port)
            self.sock.connect((ip_address, port))
            # Send ID
            self.sync_send_labeled_data(data_to_send=instance_id, label="instance_ID", receiver=self.sock, send_read_command=False)
        # Flag to trigger client's shutdown
        self.close_client: bool = False

    def initialize(self) -> None:
        """
        Run __initialize method with asyncio.

        :return:
        """

        run(self.__initialize())

    async def __initialize(self) -> None:
        """
        Receive parameters from the server to create environment, send parameters to the server in exchange.

        :return:
        """

        loop = get_event_loop()

        # Receive parameters
        recv_param_dict = {}
        await self.receive_dict(recv_to=recv_param_dict, sender=self.sock, loop=loop)
        # Use received parameters
        self.recv_parameters(recv_param_dict)

        # Create the environment
        self.create()
        self.init()

        # Send visualization
        visu_dict = self.send_visualization()
        await self.send_visualization_data(visualization_data=visu_dict, loop=loop, receiver=self.sock)

        # Send parameters
        param_dict = self.send_parameters()
        await self.send_dict(name="parameters", dict_to_send=param_dict, loop=loop, receiver=self.sock)

        # Initialization done
        await self.send_command_done(loop=loop, receiver=self.sock)

    def launch(self) -> None:
        """
        Run __launch method with asyncio.

        :return:
        """

        run(self.__launch())

    async def __launch(self) -> None:
        """
        Trigger the main communication protocol with the server.

        :return:
        """

        try:
            # Run the communication protocol with server while Client is not asked to shut down
            while not self.close_client:
                await self.__communicate(server=self.sock)
        except KeyboardInterrupt:
            print(f"[{self.name}] KEYBOARD INTERRUPT: CLOSING PROCEDURE")
        finally:
            # Closing procedure when Client is asked to shut down
            await self.__close()

    async def __communicate(self, server: socket) -> None:
        """
        Communication protocol with a server. First receive a command from the client, then process the appropriate
        actions.

        :param server: TcpIpServer to communicate with
        :return:
        """

        loop = get_event_loop()
        await self.listen_while_not_done(loop=loop, sender=server, data_dict={})

    async def __close(self) -> None:
        """
        Close the environment and shutdown the client.

        :return:
        """

        # Close environment
        self.close()
        # Confirm exit command to the server
        loop = get_event_loop()
        await self.send_command_exit(loop=loop, receiver=self.sock)
        # Close socket
        self.sock.close()

    async def send_training_data(self, network_input: Optional[numpy.ndarray] = None, network_output: Optional[numpy.ndarray] = None, loop: Any = None, receiver: socket = None) -> None:
        """
        Send the training data to the TcpIpServer.

        :param ndarray network_input: data to send under the label 'input'
        :param ndarray network_output: data to send under the label 'output'
        :param loop: get_event_loop() return
        :param receiver: TcpIpObject receiver
        :return:
        """

        loop = get_event_loop() if loop is None else loop
        receiver = self.sock if receiver is None else receiver
        # TODO: TcpIpServer no longer check the validity of the computed sample, do from TcpIpClient to compute another
        # Send network input
        if network_input is not None:
            await self.send_labeled_data(data_to_send=network_input, label="input", loop=loop, receiver=receiver)
        # Send network output
        if network_output is not None:
            await self.send_labeled_data(data_to_send=network_output, label="output", loop=loop, receiver=receiver)
        # Send additional input data
        for key in self.additional_inputs.keys():
            await self.send_labeled_data(data_to_send=self.additional_inputs[key], label='dataset_in'+key,
                                         loop=loop, receiver=receiver)
        # Send additional output data
        for key in self.additional_outputs.keys():
            await self.send_labeled_data(data_to_send=self.additional_outputs[key], label='dataset_out'+key,
                                         loop=loop, receiver=receiver)

    def sync_send_training_data(self, network_input: Optional[numpy.ndarray] = None, network_output: Optional[numpy.ndarray] = None, receiver: socket = None) -> None:
        """
        Send the training data to the TcpIpServer.
        Synchronous version of 'TcpIpClient.send_training_data'.

        :param ndarray network_input: data to send under the label 'input'
        :param ndarray network_output: data to send under the label 'output'
        :param receiver: TcpIpObject receiver
        :return:
        """

        receiver = self.sock if receiver is None else receiver
        # TODO: TcpIpServer no longer check the validity of the computed sample, do from TcpIpClient to compute another
        # check = self.check_sample()
        # self.sync_send_labeled_data(data_to_send=check, label="check", receiver=receiver)
        # Send network input
        if network_input is not None:
            self.sync_send_labeled_data(data_to_send=network_input, label="input", receiver=receiver)
        # Send network output
        if network_output is not None:
            self.sync_send_labeled_data(data_to_send=network_output, label="output", receiver=receiver)
        # Send additional input data
        for key in self.additional_inputs.keys():
            self.sync_send_labeled_data(data_to_send=self.additional_inputs[key], label='dataset_in'+key,
                                        receiver=receiver)
        for key in self.additional_outputs.keys():
            self.sync_send_labeled_data(data_to_send=self.additional_outputs[key], label='dataset_out'+key,
                                        receiver=receiver)

    async def send_prediction_request(self, network_input: Optional[numpy.ndarray], loop: Any = None, receiver: socket = None) -> numpy.ndarray:
        """
        Request a prediction from the Environment.

        :param ndarray network_input: Data to send under the label 'input'
        :param loop: get_event_loop() return
        :param receiver: TcpIpObject receiver
        :return:
        """

        loop = get_event_loop() if loop is None else loop
        receiver = self.sock if receiver is None else receiver
        # Send prediction command
        await self.send_command_prediction()
        # Send the network input
        await self.send_labeled_data(data_to_send=network_input, label='input', receiver=receiver)
        # Receive the network prediction
        label, pred = await self.receive_labeled_data(loop=loop, sender=receiver)
        return pred

    def sync_send_prediction_request(self, network_input: Optional[numpy.ndarray], receiver: socket = None) -> numpy.ndarray:

        """
        Request a prediction from the Environment.
        Synchronous version of 'TcpIpClient.send_prediction_request'.

        :param ndarray network_input: Data to send under the label 'input'
        :param receiver: TcpIpObject receiver
        :return:
        """

        receiver = self.sock if receiver is None else receiver
        # Send prediction command
        self.sync_send_command_prediction()
        # Send the network input
        self.sync_send_labeled_data(data_to_send=network_input, label='input', receiver=receiver)
        # Receive the network prediction
        _, pred = self.sync_receive_labeled_data()
        return pred

    async def send_visualization_data(self, visualization_data: Dict[Any, Any] = None, loop: Any = None, receiver: socket = None) -> None:
        """
        Send the visualization data to TcpIpServer.

        :param dict visualization_data: Updated visualization data.
        :param loop: get_event_loop() return
        :param receiver: TcpIpObject receiver
        :return:
        """

        loop = get_event_loop() if loop is None else loop
        receiver = self.sock if receiver is None else receiver
        # Send 'visualization' command
        await self.send_command_visualisation()
        # Send visualization data
        await self.send_dict(name="visualisation", dict_to_send=visualization_data, receiver=receiver, loop=loop)

    def sync_send_visualization_data(self, visualization_data: Dict[Any, Any] = None, receiver: socket = None) -> None:
        """
        Send the visualization data to TcpIpServer.
        Synchronous version of 'TcpIpClient.send_visualization_data'.

        :param dict visualization_data: Updated visualization data.
        :param receiver: TcpIpObject receiver
        :return:
        """

        receiver = self.sock if receiver is None else receiver
        # Send 'visualization' command
        self.sync_send_command_visualisation()
        # Send visualization data
        self.sync_send_dict(name="visualisation", dict_to_send=visualization_data, receiver=receiver)

    async def action_on_compute(self, data: numpy.ndarray, client_id: int, sender: socket, loop: Any) -> None:
        """
        Action to run when receiving the 'compute' command

        :param dict data: Dict storing data
        :param int client_id: ID of the TcpIpClient
        :param loop: asyncio.get_event_loop() return
        :param sender: TcpIpObject sender
        :return:
        """
        # Compute data flag set to True
        self.compute_essential_data = True
        # Trigger a step
        await self.step()
        # Protocol done
        await self.send_command_done()

    async def action_on_exit(self, data: numpy.ndarray, client_id: int, sender: socket, loop: Any) -> None:
        """
        Action to run when receiving the 'exit' command

        :param dict data: Dict storing data
        :param int client_id: ID of the TcpIpClient
        :param loop: asyncio.get_event_loop() return
        :param sender: TcpIpObject sender
        :return:
        """
        # Close client flag set to True
        self.close_client = True

    async def action_on_prediction(self, data: numpy.ndarray, client_id: int, sender: socket, loop: Any) -> None:
        """
        Action to run when receiving the 'prediction' command

        :param dict data: Dict storing data
        :param int client_id: ID of the TcpIpClient
        :param loop: asyncio.get_event_loop() return
        :param sender: TcpIpObject sender
        :return:
        """
        # Receive prediction
        prediction = await self.receive_data(loop=loop, sender=sender)
        # Apply the prediction in Environment
        self.apply_prediction(prediction)

    async def action_on_sample(self, data: numpy.ndarray, client_id: int, sender: socket, loop: Any) -> None:
        """
        Action to run when receiving the 'sample' command

        :param dict data: Dict storing data
        :param int client_id: ID of the TcpIpClient
        :param loop: asyncio.get_event_loop() return
        :param sender: TcpIpObject sender
        :return:
        """
        # Receive input sample
        sample_in = await self.receive_data(loop=loop, sender=sender)
        # Receive output sample
        sample_out = await self.receive_data(loop=loop, sender=sender)

        additional_in, additional_out = {}, {}
        # Receive additional input sample if there are any
        if await self.receive_data(loop=loop, sender=sender):
            additional_in = {}
            await self.receive_dict(recv_to=additional_in, loop=loop, sender=sender)
        # Receive additional output sample if there are any
        if await self.receive_data(loop=loop, sender=sender):
            additional_out = {}
            await self.receive_dict(recv_to=additional_out, loop=loop, sender=sender)

        # Set the samples from Dataset
        self.set_dataset_sample(sample_in, sample_out, additional_in, additional_out)

    async def action_on_step(self, data: numpy.ndarray, client_id: int, sender: socket, loop: Any) -> None:
        """
        Action to run when receiving the 'step' command

        :param dict data: Dict storing data
        :param int client_id: ID of the TcpIpClient
        :param loop: asyncio.get_event_loop() return
        :param sender: TcpIpObject sender
        :return:
        """

        # Compute data flag set to False
        self.compute_essential_data = False
        # Trigger a step
        await self.step()
        # Done protocol
        await self.send_command_done()
