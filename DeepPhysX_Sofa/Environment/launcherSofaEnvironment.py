import os
import sys
import Sofa.Simulation

from DeepPhysX_Sofa.Environment.SofaEnvironment import SofaEnvironment as Environment
from DeepPhysX_Core.AsyncSocket.BytesBaseConverter import BytesBaseConverter as Converter

if __name__ == '__main__':

    # Check script call
    if len(sys.argv) != 7:
        print(f"Usage: python3 {sys.argv[0]} <file_path> <environment_class> <ip_address> <port> "
              f"<converter_class> <idx>")
        sys.exit(1)

    # Import environment_class
    sys.path.append(os.path.dirname(sys.argv[1]))
    module_name = sys.argv[1].split(os.sep)[-1][:-3]
    exec(f"from {module_name} import {sys.argv[2]} as Environment")

    # Import converter_class
    exec(f"from DeepPhysX_Core.AsyncSocket.{sys.argv[5]} import {sys.argv[5]} as Converter")

    # Create root node
    root_node = Sofa.Core.Node('rootNode')

    # Create, init and run Tcp-Ip environment
    client = root_node.addObject(Environment(ip_address=sys.argv[3], port=int(sys.argv[4]), data_converter=Converter,
                                             instance_id=int(sys.argv[6]), root_node=root_node))

    client.initialize()
    client.run()

    print(f"[launcherBaseEnvironment] Shutting down client {sys.argv[3]}")
