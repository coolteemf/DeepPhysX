import sys
import Sofa.Simulation

from DeepPhysX_Core.AsyncSocket.BytesBaseConverter import BytesBaseConverter as Converter
import os

if __name__ == '__main__':

    # Check script call
    if len(sys.argv) != 7:
        print(f"Usage: python3 {sys.argv[0]} <file_path> <environment_class> <ip_address> <port> "
              f"<converter_class> <idx>")
        sys.exit(1)

    # Import environment_class
    sys.path.append(os.path.dirname(sys.argv[1]))
    module_name = sys.argv[1].split(os.sep)[-1][:-3]
    # Import converter_class
    exec(f"from {module_name} import {sys.argv[2]} as Env")
    # Create root node
    root_node = Sofa.Core.Node('rootNode')

    # Create, init and run Tcp-Ip environment
    client = root_node.addObject(Env(ip_address=sys.argv[3], port=int(sys.argv[4]), data_converter=Converter,
                                             instance_id=int(sys.argv[6]), root_node=root_node))
    client.initialize()
    client.run()

    print(f"[launcherBaseEnvironment] Shutting down client {sys.argv[3]}")
