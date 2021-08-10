import sys
import numpy as np
import time
import asyncio

# import Sofa
# import Sofa.Simulation
# from Environment import Environment

from DeepPhysX_Core.AsyncSocket.TcpIpObject import TcpIpObject, BytesBaseConverter
from DeepPhysX_Core.Environment.BaseEnvironment import BaseEnvironment as Environment
from BytesNumpyConverter import BytesNumpyConverter

if len(sys.argv) != 4:
    print(f"Usage: python3 TcpIpClient.py <file_path> <environment_class> <idx>")
    sys.exit(1)

sys.path.append(sys.argv[1])
exec("from "+sys.argv[2]+" import "+sys.argv[2]+" as Environment")


class TcpIpClient(TcpIpObject):

    def __init__(self, ip_address='localhost', port=10000, data_converter=BytesBaseConverter, idx=1):
        super(TcpIpClient, self).__init__(ip_address=ip_address, port=port, data_converter=data_converter)
        self.sock.connect((ip_address, port))

        self.bytes_field_to_send = None
        self.bytes_field_received = None
        self.close_server = False
        self.data_size = 0

        # root = Sofa.Core.Node('root')
        # self.environment = root.addObject(Environment(root=root, idx=idx))
        # Sofa.Simulation.init(root)
        print("Create env")
        self.environment = Environment(config=None)

        time.sleep(1)
        asyncio.run(self.run())

    def generate_data(self):
        self.environment.step()
        return self.environment.getTensor()

    async def run(self):
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
        loop = asyncio.get_event_loop()
        command = await self.receive_data(loop, server, expect_command=True)
        if command not in [b'exit', b'wait', b'step']:
            raise ValueError("Unknown command")

        if command == b'exit':
            self.close_server = True

        elif command == b'wait':
            await self.send_wait_command(loop=loop, receiver=server)

        elif command == b'step':
            # Get a tensor
            data = self.generate_data()
            data_shape = data.shape

            # Send the tensor to server
            await self.send_data(data_to_send=data, loop=loop, receiver=server)

            # Retrieve data from server
            data_from_server = await self.receive_data(loop=loop, sender=server)

            # Check if both data are the same
            if np.linalg.norm(data-data_from_server.reshape(data_shape), axis=None):
                raise ValueError(f"Received data mismatches sent data in {self.environment.idx}:"
                                 f"{np.linalg.norm(data-data_from_server, axis=None)}.")


if __name__ == '__main__':
    print(sys.argv[2])
    main_client = TcpIpClient(data_converter=BytesNumpyConverter, idx=int(sys.argv[3]))
    del main_client
    print("Shutting down client", sys.argv[1])
