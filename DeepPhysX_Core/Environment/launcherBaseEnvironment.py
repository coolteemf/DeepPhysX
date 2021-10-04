from os import sep
from os.path import dirname
from sys import argv, path
from DeepPhysX_Core.AsyncSocket.BytesBaseConverter import BytesBaseConverter as Converter

if __name__ == '__main__':

    # Check script call
    if len(argv) != 8:
        print(f"Usage: python3 {argv[0]} <file_path> <environment_class> <ip_address> <port> "
              f"<converter_class> <instance_id> <max_instance_count>")
        exit(1)

    # Import environment_class
    path.append(dirname(argv[1]))
    module_name = argv[1].split(sep)[-1][:-3]
    exec(f"from {module_name} import {argv[2]} as Environment")

    # Import converter_class
    exec(f"from DeepPhysX_Core.AsyncSocket.{argv[5]} import {argv[5]} as Converter")

    # Create, init and run Tcp-Ip environment
    client = Environment(ip_address=argv[3], port=int(argv[4]), data_converter=Converter,
                         instance_id=int(argv[6]), number_of_instances=int(argv[7]))

    print("INIT DONE")
    client.initialize()
    client.launch()

    print(f"[launcherBaseEnvironment] Shutting down client {argv[3]}")
