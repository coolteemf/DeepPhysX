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
# from DeepPhysX_Core.AsyncSocket.AbstractEnvironment import AbstractEnvironment
# from DeepPhysX_Core.AsyncSocket.BytesBaseConverter import BytesBaseConverter
# from DeepPhysX_Core.AsyncSocket.BytesNumpyConverter import BytesNumpyConverter
# from DeepPhysX_Core.AsyncSocket.TcpIpClient import TcpIpClient
# from DeepPhysX_Core.AsyncSocket.TcpIpObject import TcpIpObject
# from DeepPhysX_Core.AsyncSocket.TcpIpServer import TcpIpServer
#
# __all__ = ["AbstractEnvironment",
#            "BytesNumpyConverter",
#            "BytesBaseConverter",
#            "TcpIpObject",
#            "TcpIpServer",
#            "TcpIpClient"]