# # from os import listdir
# # from os.path import isdir, isfile, join, sep
# # from pathlib import Path
# #
# # current_absolute_path = Path(__file__).parent.absolute()
# # current_relative_directory = str(Path(__file__).parent).split(sep)[-1]
# #
# # for object in listdir(current_absolute_path):
# #     # Ignore current and any __pycache__ directories
# #     if isdir(object) and not "__pycache__" in object:
# #         exec(f"from {current_relative_directory} import {object}")
# #     # Import python file that is not this one
# #     elif isfile(object) and ".py" in object and not "__init__.py" == object:
# #         exec(f"import {object}")
#
# from DeepPhysX_Core.Network.BaseNetwork import BaseNetwork
# from DeepPhysX_Core.Network.BaseNetworkConfig import BaseNetworkConfig
# from DeepPhysX_Core.Network.BaseOptimization import BaseOptimization
# from DeepPhysX_Core.Network.DataTransformation import DataTransformation
#
# __all__ = ["BaseNetwork",
#            "BaseNetworkConfig",
#            "BaseOptimization",
#            "DataTransformation"]