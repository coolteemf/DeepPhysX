import sys
import socket
import numpy as np
import time
import asyncio

import Sofa
import Sofa.Simulation
from Environment import Environment


class TcpIpClient:

    def __init__(self, ip_address='localhost', port=10000, idx=1):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.connect((ip_address, port))
        self.bytes_field_to_send = None
        self.bytes_field_received = None
        self.close_server = False
        self.data_size = 0

        root = Sofa.Core.Node('root')
        self.environment = root.addObject(Environment(root=root, idx=idx))
        Sofa.Simulation.init(root)

        time.sleep(1)
        self.run()

    def generate_data(self):
        self.environment.step()
        print(self.environment)
        return self.environment.getTensor()

    def bytes_to_data(self, bytes_field):
        return np.frombuffer(bytes_field)

    def data_to_bytes(self, data):
        return data.tobytes()

    def run(self):
        try:
            #Send data size for communications
            self.data_size = self.generate_data().nbytes
            self.server.sendall(self.data_size.to_bytes(4, byteorder='big'))

            while not self.close_server:
                self.discuss()
            # print("")
        except KeyboardInterrupt:
            print("KEYBOARD INTERRUPT: CLOSING PROCEDURE")
        finally:
            self.server.send(bytes('exit', 'utf-8'))
            self.server.close()

    def discuss(self):
        command = self.server.recv(5)
        if command not in [b'exit', b'wait', b'step']:
            raise ValueError("Unknown command")
        if command == b'exit':
            # print(f"Client {self.environment.idx} received command 'exit")
            self.close_server = True
        elif command == b'wait':
            # print(f"Client {self.environment.idx} received command 'wait")
            self.server.send(bytes('wait', 'utf-8'))
        elif command == b'step':
            # print(f"Client {self.environment.idx} received command 'step")
            data = self.generate_data()
            # Todo: add flag to check if data size in bytes has been initialized (instead of putting 255)
            self.bytes_field_to_send = self.data_to_bytes(data)
            self.server.send(self.bytes_field_to_send)
            self.bytes_field_received = self.server.recv(self.data_size)
            if self.bytes_field_received != self.bytes_field_to_send:
                raise ValueError(f"Received data mismatches sent data in {self.environment.idx}:"
                                 f"{data} != {self.bytes_to_data(self.bytes_field_received)}.")


if __name__ == '__main__':
    client = TcpIpClient(idx=int(sys.argv[1]))
    del client
    print("Shutting down client", sys.argv[1])
