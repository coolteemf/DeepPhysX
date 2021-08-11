import sys
import numpy as np
import asyncio

# import Sofa
# import Sofa.Simulation
# from Environment import Environment

from DeepPhysX_Core.AsyncSocket.TcpIpObject import TcpIpObject, BytesNumpyConverter
from DeepPhysX_Core.Environment.BaseEnvironment import BaseEnvironment as Environment

if len(sys.argv) != 4:
    print(f"Usage: python3 TcpIpClient.py <file_path> <environment_class> <idx>")
    sys.exit(1)

sys.path.append(sys.argv[1])
exec("from "+sys.argv[2]+" import "+sys.argv[2]+" as Environment")


class TcpIpClient(TcpIpObject):

    def __init__(self, ip_address='localhost', port=10000, data_converter=BytesNumpyConverter, idx=1):
        """

        :param ip_address:
        :param port:
        :param data_converter:
        :param idx:
        """
        super(TcpIpClient, self).__init__(ip_address=ip_address, port=port, data_converter=data_converter)
        self.sock.connect((ip_address, port))

        self.bytes_field_to_send = None
        self.bytes_field_received = None
        self.close_server = False
        self.data_size = 0

        # root = Sofa.Core.Node('root')
        # self.environment = root.addObject(Environment(root=root, idx=idx))
        # Sofa.Simulation.init(root)
        self.environment = Environment(idx)
        self.environment.create()

        asyncio.run(self.run())

    async def run(self):
        """

        :return:
        """
        loop = asyncio.get_event_loop()
        try:
            while not self.close_server:
                await self.communicate(server=self.sock)
        except KeyboardInterrupt:
            print("KEYBOARD INTERRUPT: CLOSING PROCEDURE")
        finally:
            await self.send_exit_command(loop=loop, receiver=self.sock)
            self.sock.close()

    async def communicate(self, client=None, server=None, id=None):
        """

        :param client:
        :param server:
        :param id:
        :return:
        """
        loop = asyncio.get_event_loop()
        command = await self.receive_data(loop, server, expect_command=True)
        # Check command existence
        if command not in [b'exit', b'step', b'test', b'size', b'c_in', b'c_ou',  b'g_in', b'g_ou']:
            raise ValueError(f"Unknown command {command}")
        # Exit
        if command == b'exit':
            self.close_server = True
        # Trigger a step in the environment
        elif command == b'step':
            self.environment.step()
        # Check if the sample is exploitable
        elif command == b'test':
            check = b'1' if self.environment.checkSample() else b'0'
            await self.send_data(data_to_send=check, loop=loop, receiver=server, do_convert=False)
        # Send the data sizes as data will be flatten in socket
        elif command == b'size':
            await self.send_data(data_to_send=np.array(self.environment.input_size, dtype=float), loop=loop, receiver=server)
            await self.send_data(data_to_send=np.array(self.environment.output_size, dtype=float), loop=loop, receiver=server)
        # Compute the environment input
        elif command == b'c_in':
            self.environment.computeInput()
        # Compute the environment output
        elif command == b'c_ou':
            self.environment.computeOutput()
        # Send the environment input
        elif command == b'g_in':
            data_in = self.environment.getInput()
            await self.send_data(data_to_send=data_in, loop=loop, receiver=server)
        # Send the environment output
        elif command == b'g_ou':
            data_out = self.environment.getOutput()
            await self.send_data(data_to_send=data_out, loop=loop, receiver=server)


if __name__ == '__main__':
    main_client = TcpIpClient(idx=int(sys.argv[3]))
    print("Shutting down client", sys.argv[3])
