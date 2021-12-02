from pathlib import Path
from os import walk
from os.path import sep
current_absolute_path = Path(__file__).parent.absolute()

for rootdir, dirs, files in walk(current_absolute_path):
    if "__pycache__" not in rootdir and "Example" not in rootdir and files != ["__init__.py"]:
        for file in files:
            if "pyc" not in file and file != "__init__.py":
                splitted_path = str(rootdir).split(sep)
                main_lib = splitted_path[-2]        # DeepPhysX_Sofa
                if "Example" in main_lib:
                    continue
                sub_lib = splitted_path[-1]         # Pipeline, utils, Network...
                filename = str(file).split(".")[0]  # BaseNetwork, TcpIpClient, DatasetManager...
                exec(f"from {main_lib}.{sub_lib}.{filename} import *")