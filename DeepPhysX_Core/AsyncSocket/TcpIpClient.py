from asyncio import run, get_event_loop
from numpy import array

from DeepPhysX_Core.AsyncSocket.TcpIpObject import TcpIpObject
from DeepPhysX_Core.AsyncSocket.AbstractEnvironment import AbstractEnvironment


class TcpIpClient(TcpIpObject, AbstractEnvironment):

    def __init__(self,
                 instance_id=1,
                 number_of_instances=1,
                 as_tcpip_client=True,
                 ip_address='localhost',
                 port=10000):
        """
        TcpIpClient is a TcpIpObject which communicate with a TcpIpServer and an AbstractEnvironment to compute data.

        :param int instance_id: ID of the instance
        :param int number_of_instances: Number of simultaneously launched instances
        :param as_tcpip_client: Environment is a TcpIpObject if True, is owned by an EnvironmentManager if False
        :param str ip_address: IP address of the TcpIpObject
        :param int port: Port number of the TcpIpObject
        """

        AbstractEnvironment.__init__(self,
                                     instance_id=instance_id,
                                     number_of_instances=number_of_instances,
                                     as_tcpip_client=as_tcpip_client)
        # Bind to client address
        if self.as_tcpip_client:
            TcpIpObject.__init__(self,
                                 ip_address=ip_address,
                                 port=port)
            self.sock.connect((ip_address, port))
            self.sync_send_labeled_data(data_to_send=instance_id, label="instance_ID", receiver=self.sock, send_read_command=False)
        # Flag to trigger client's shutdown
        self.close_client = False

    def initialize(self):
        """
        Run __initialize method with asyncio.

        :return:
        """
        run(self.__initialize())

    async def __initialize(self):
        """
        Receive parameters from the server to create environment, send parameters to the server in exchange.

        :return:
        """
        loop = get_event_loop()

        recv_param_dict = {}
        # Receive parameters
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
<<<<<<< HEAD
<<<<<<< HEAD
        for key in param_dict:
            # Send the parameter (label + data)
            await self.send_labeled_data(data_to_send=param_dict[key], label=key, loop=loop, receiver=self.sock)
            # Tell the client to stop receiving data
        await self.send_command_done(loop=loop, receiver=self.sock)

        # Receive and check last init server command
        cmd = await self.receive_data(loop, self.sock)
        if cmd not in [b'done', b'size']:
            raise ValueError(f"[{self.name}] Unknown command {cmd}, must be in {[b'done', b'size']}")
        # If server asked to send IO sizes
        if cmd == b'size':
            # Send input and output sizes
            await self.send_data(data_to_send=array(self.input_size, dtype=float), loop=loop,
                                 receiver=self.sock)
            await self.send_data(data_to_send=array(self.output_size, dtype=float), loop=loop,
                                 receiver=self.sock)
=======
        print("CLIENT __initialize PARAM")
        await self.send_dict(name="parameters", dict_to_send=param_dict, loop=loop, receiver=self.sock)

        await self.send_command_done(loop=loop, receiver=self.sock)
>>>>>>> Server and client handle multiple object types and dictionaries
=======
        await self.send_dict(name="parameters", dict_to_send=param_dict, loop=loop, receiver=self.sock)

        # I don't know why but this if is needed
        if visu_dict not in [{}, None] or param_dict not in [{}, None]:
            await self.send_command_done(loop=loop, receiver=self.sock)
>>>>>>> TCPIP Works with and without visualizer.

    def launch(self):
        """
        Run __launch method with asyncio.

        :return:
        """
        run(self.__launch())

    async def __launch(self):
        """
        Trigger the main communication protocol with the server.

        :return:
        """
        try:
            # Run the communication protocol with server while client is not asked to shutdown
            while not self.close_client:
                await self.__communicate(server=self.sock)
        except KeyboardInterrupt:
            print(f"[{self.name}] KEYBOARD INTERRUPT: CLOSING PROCEDURE")
        finally:
            await self.__close()

    async def __communicate(self, server=None):
        """
        Communication protocol with a server. First receive a command from the client, then process the appropriate
        actions.

        :param server: TcpIpServer to communicate with
        :return:
        """
        loop = get_event_loop()
<<<<<<< HEAD

        # Receive and check command
        command = await self.receive_data(loop, server)
        if command not in self.command_dict.values():
            raise ValueError(f"[{self.name}] Unknown command {command}")

        # 'exit': close the environment and the client
        if command == b'exit':
            self.close_client = True
        # 'step': trigger a step in the environment
        elif command == b'step':
            self.compute_essential_data = False
            await self.step()
            self.sync_send_command_done()
        # 'cmpt': trigger a step in the environment and call for the sending of the training data
        elif command == b'cmpt':
            self.compute_essential_data = True
            await self.step()
            self.sync_send_command_done()
        # 'test': check if the sample is exploitable
        elif command == b'test':
            check = b'1' if self.checkSample() else b'0'
            await self.send_data(data_to_send=check, loop=loop, receiver=server)
        # 'pred': receive prediction and apply it
        elif command == b'pred':
            prediction = await self.receive_data(loop=loop, sender=server)
            self.applyPrediction(prediction.reshape(self.output_size))
        # 'samp': receive a sample from Dataset
        elif command == b'samp':
            sample_in = await self.receive_data(loop=loop, sender=server)
            sample_out = await self.receive_data(loop=loop, sender=server)
            additional_in, additional_out = None, None
            # Is there other in fields ?
            if await self.receive_data(loop=loop, sender=server):
                additional_in = await self.receive_dict_data(loop=loop, sender=server)
            # Is there other out fields ?
            if await self.receive_data(loop=loop, sender=server):
                additional_out = await self.receive_dict_data(loop=loop, sender=server)
            self.setDatasetSample(sample_in, sample_out, additional_in, additional_out)
=======
        await self.listen_while_not_done(loop=loop, sender=server, data_dict={}, client_id=None)
>>>>>>> Server and client handle multiple object types and dictionaries

    async def __close(self):
        """
        Close the environment and shutdown the client.

        :return:
        """
        # Close environment
        self.close()

        # Shutdown client
        loop = get_event_loop()
        # Confirm exit command to the server
        await self.send_command_exit(loop=loop, receiver=self.sock)
        # Close socket
        self.sock.close()

    async def listen_while_not_done(self, loop, sender, data_dict):
        """
        Listening to data until command 'done' is received.

        :param loop: asyncio.get_event_loop() return
        :param sender: TcpIpObject sender
        :param data_dict: Dictionary in which data is stored
        :return:
        """
        while await self.receive_data(loop=loop, sender=sender) != self.command_dict['done']:
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

    def sync_listen_while_not_done(self, data_dict):
        """
        Listening to data until command 'done' is received.

        :param data_dict: Dictionary in which data is stored
        :return:
        """
        while self.sync_receive_data() != self.command_dict['done']:
            label, param = self.sync_receive_labeled_data()
            data_dict[label] = param

    async def send_training_data(self, network_input=None, network_output=None, loop=None, receiver=None):
        """
        Send the training data to the TcpIpServer.

        :param loop: get_event_loop() return
        :param receiver: TcpIpObject receiver
        :param network_input: data to send under the label \"input\"
        :param network_output: data to send under the label \"output\"
        :return:
        """
        loop = get_event_loop() if loop is None else loop
        receiver = self.sock if receiver is None else receiver
<<<<<<< HEAD
        # Check the validity of the computed sample
        check = self.checkSample()
        await self.send_labeled_data(data_to_send=check, label="check", loop=loop, receiver=receiver)
        # Send training data if sample is valid
        if check:
            # Send network input
            if network_input is not None:
                await self.send_labeled_data(data_to_send=network_input, label="input", loop=loop, receiver=receiver)
            # Send network output
            if network_output is not None:
                await self.send_labeled_data(data_to_send=network_output, label="output", loop=loop, receiver=receiver)
            # Send additional data
            for key in self.additional_inputs.keys():
                await self.send_labeled_data(data_to_send=self.additional_inputs[key], label='dataset_in'+key,
                                             loop=loop, receiver=receiver)
            for key in self.additional_outputs.keys():
                await self.send_labeled_data(data_to_send=self.additional_outputs[key], label='dataset_out'+key,
                                             loop=loop, receiver=receiver)
=======
        if network_input is not None:
            await self.send_labeled_data(data_to_send=network_input, label="input", loop=loop, receiver=receiver)
        if network_output is not None:
            await self.send_labeled_data(data_to_send=network_output, label="output", loop=loop, receiver=receiver)
>>>>>>> Server and client handle multiple object types and dictionaries

    async def send_prediction_request(self, network_input=None, loop=None, receiver=None):
        """

        :param loop:
        :param network_input: Data to send under the label 'input'
        :param receiver: TcpIpObject receiver
        :return:
        """
        loop = get_event_loop() if loop is None else loop
        receiver = self.sock if receiver is None else receiver
        if network_input is not None:
            await self.send_command_prediction()
            await self.send_labeled_data(data_to_send=network_input, label='input', receiver=receiver)
            label, pred = await self.receive_labeled_data(loop=loop, sender=receiver)
            return pred

    def sync_send_training_data(self, network_input=None, network_output=None, receiver=None):
        """
        Send the training data to the TcpIpServer.

        :param receiver: TcpIpObject receiver
        :param network_input: data to send under the label \"input\"
        :param network_output: data to send under the label \"output\"
        :return:
        """
        receiver = self.sock if receiver is None else receiver
<<<<<<< HEAD
        # Check the validity of the computed sample
        check = self.checkSample()
        self.sync_send_labeled_data(data_to_send=check, label="check", receiver=receiver)
        # Send training data if sample is valid
        if check:
            # Send network input
            if network_input is not None:
                self.sync_send_labeled_data(data_to_send=network_input, label="input", receiver=receiver)
            # Send network output
            if network_output is not None:
                self.sync_send_labeled_data(data_to_send=network_output, label="output", receiver=receiver)
            # Send additional data
            for key in self.additional_inputs.keys():
                self.sync_send_labeled_data(data_to_send=self.additional_inputs[key], label='dataset_in'+key,
                                            receiver=receiver)
            for key in self.additional_outputs.keys():
                self.sync_send_labeled_data(data_to_send=self.additional_outputs[key], label='dataset_out'+key,
                                            receiver=receiver)
=======
        if network_input is not None:
            self.sync_send_command_read()
            self.sync_send_labeled_data(data_to_send=network_input, label="input", receiver=receiver)
        if network_output is not None:
            self.sync_send_command_read()
            self.sync_send_labeled_data(data_to_send=network_output, label="output", receiver=receiver)
>>>>>>> Server and client handle multiple object types and dictionaries

    def sync_send_prediction_request(self, network_input=None, receiver=None):
        """
        Request a prediction from the Environment.

        :param network_input: Data to send under the label 'input'
        :param receiver: TcpIpObject receiver
        :return:
        """
        receiver = self.sock if receiver is None else receiver
        if network_input is not None:
            # Send related command
            self.sync_send_command_prediction()
            # Send the network input
            self.sync_send_labeled_data(data_to_send=network_input, label='input', receiver=receiver)
            # Receive the network prediction
            label, pred = self.sync_receive_labeled_data()
            return pred

    def sync_send_visualization_data(self, visualization_data=None, receiver=None):
<<<<<<< HEAD
<<<<<<< HEAD
        """
        Send the visualization data to the TcpIpServer.

        :param visualization_data: Dict containing the data fields to update the visualization
        :param receiver: TcpIpObject receiver
        :return:
        """
        receiver = self.sock if receiver is None else receiver
        if visualization_data is not None:
            # Send related command
            self.sync_send_command_visualization()
            # Send each field in the data dict
            for key in visualization_data:
                self.sync_send_labeled_data(data_to_send=visualization_data[key], label=key, receiver=receiver)
            # Done sending visualization data
            self.sync_send_command_done(receiver=self.sock)
=======
=======
        self.sync_send_command_visualisation()
>>>>>>> TCPIP Works with and without visualizer.
        self.sync_send_dict(name="visualisation", dict_to_send=visualization_data)

    async def send_visualization_data(self, visualization_data=None, loop=None, receiver=None):
        loop = get_event_loop() if loop is None else loop
        receiver = self.sock if receiver is None else receiver
        await self.send_command_visualisation()
        # print("CLIENT send visu")
        # print(visualization_data)
        await self.send_dict(name="visualisation", dict_to_send=visualization_data, receiver=receiver, loop=loop)
        #print("CLIENT VISU SENT")

    async def action_on_exit(self, data, client_id, sender=None, loop=None):
        self.close_client = True

    async def action_on_step(self, data, client_id, sender=None, loop=None):
        self.compute_essential_data = False
        await self.step()
        await self.send_command_done()
        # I don't know why this second line is needed probably fuckedup somewhere
        await self.send_command_done()


    async def action_on_done(self, data, client_id, sender=None, loop=None):
        pass

    async def action_on_prediction(self, data, client_id, sender=None, loop=None):
        prediction = await self.receive_data(loop=loop, sender=sender)
        self.applyPrediction(prediction.reshape(self.output_size))

    async def action_on_compute(self, data,  client_id, sender=None, loop=None):
        self.compute_essential_data = True
        await self.step()
        await self.send_command_done()
        # I don't know why this second line is needed probably fuckedup somewhere
        await self.send_command_done()

    async def action_on_sample(self, data, client_id, sender=None, loop=None):
        sample_in = await self.receive_data(loop=loop, sender=sender)
        sample_out = await self.receive_data(loop=loop, sender=sender)
        self.setDatasetSample(sample_in, sample_out)
>>>>>>> Server and client handle multiple object types and dictionaries
