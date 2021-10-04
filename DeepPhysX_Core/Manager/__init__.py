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
# from DeepPhysX_Core.Manager.DataManager import DataManager
# from DeepPhysX_Core.Manager.DatasetManager import DatasetManager
# from DeepPhysX_Core.Manager.EnvironmentManager import EnvironmentManager
# from DeepPhysX_Core.Manager.Manager import Manager
# from DeepPhysX_Core.Manager.NetworkManager import NetworkManager
# from DeepPhysX_Core.Manager.StatsManager import StatsManager
# from DeepPhysX_Core.Manager.VisualizerManager import VisualizerManager
#
# __all__ = ["DataManager",
#            "DatasetManager",
#            "EnvironmentManager",
#            "Manager",
#            "NetworkManager",
#            "StatsManager",
#            "VisualizerManager"]