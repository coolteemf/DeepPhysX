from os import sep
from os.path import dirname
from sys import argv, exit, path

from DeepPhysX_Core.Environment.BaseEnvironment import BaseEnvironment as Environment
from DeepPhysX_Core.AsyncSocket.BytesBaseConverter import BytesBaseConverter as Converter

if __name__ == '__main__':

    # Check script call
    if len(argv) != 8:
        print(f"Usage: python3 {argv[0]} <file_path> <environment_class> <ip_address> <port> "
              f"<converter_class> <client_id> <max_number_of_instance>")
        exit(1)

    # Import environment_class
    path.append(dirname(argv[1]))
    module_name = argv[1].split(sep)[-1][:-3]
    exec(f"from {module_name} import {argv[2]} as Env")

    # Import converter_class
    exec(f"from DeepPhysX_Core.AsyncSocket.{argv[5]} import {argv[5]} as Converter")

    # Create, init and run Tcp-Ip environment
    client = Environment(ip_address=argv[3], port=int(argv[4]), data_converter=Converter,
                         instance_id=int(argv[6]))
    client.initialize()
    client.listen_server()

    print(f"[launcherBaseEnvironment] Shutting down client {argv[3]}")
