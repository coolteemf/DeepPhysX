from os import sep
from os.path import dirname
from sys import argv, path

if __name__ == '__main__':

    # Check script call
    if len(argv) != 7:
        print(f"Usage: python3 {argv[0]} <file_path> <environment_class> <ip_address> <port> "
              f"<instance_id> <max_instance_count>")
        exit(1)

    # Import environment_class
    path.append(dirname(argv[1]))
    module_name = argv[1].split(sep)[-1][:-3]
    exec(f"from {module_name} import {argv[2]} as Environment")

    # Create, init and run Tcp-Ip environment
    client = Environment(ip_address=argv[3], port=int(argv[4]), instance_id=int(argv[5]),
                         number_of_instances=int(argv[6]))

    print("INIT DONE")
    client.initialize()
    client.launch()

    print(f"[launcherBaseEnvironment] Shutting down client {argv[3]}")
