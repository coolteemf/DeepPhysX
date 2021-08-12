import sys

from DeepPhysX_Core.Environment.BaseEnvironment import BaseEnvironment as Environment


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print(f"Usage: python3 launcherBaseEnvironment.py <file_path> <environment_class> <idx>")
        sys.exit(1)

    sys.path.append(sys.argv[1])
    exec("from " + sys.argv[2] + " import " + sys.argv[2] + " as Environment")

    client = Environment(instance_id=int(sys.argv[3]))
    client.initialize()
    client.run()
    print("Shutting down client", sys.argv[3])
