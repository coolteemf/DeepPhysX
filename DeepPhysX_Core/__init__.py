from pathlib import Path
from os import walk
from os.path import sep
current_absolute_path = Path(__file__).parent.absolute()

for rootdir, dirs, files in walk(current_absolute_path):
    if "__pycache__" not in rootdir and files != ["__init__.py"]:
        for file in files:
            if "pyc" not in file and file != "__init__.py":
                splitted_path = str(rootdir).split(sep)
                # additiv process to include anything below DeepPhysX_Core
                import_path = f""
                for lib in splitted_path[::-1]:
                    if 'DeepPhysX_Core' == lib:
                        filename = str(file).split(".")[0]  # BaseNetwork, TcpIpClient, DatasetManager...
                        print(f"from {lib}.{import_path}{filename} import *")
                        exec(f"from {lib}.{import_path}{filename} import *")
                        break
                    import_path = f"{lib}." + import_path
