from pathlib import Path
from os import walk
from os.path import sep
current_absolute_path = Path(__file__).parent.absolute()

for rootdir, dirs, files in walk(current_absolute_path):
    if "__pycache__" not in rootdir and files != ["__init__.py"]:
        for file in files:
            if "pyc" not in file and file != "__init__.py":
                splitted_path = str(rootdir).split(sep)
                main_lib = splitted_path[-2]        # DeepPhysX_Core
                sub_lib = splitted_path[-1]         # Pipeline, utils, Network...
                filename = str(file).split(".")[0]  # BaseNetwork, TcpIpClient, DatasetManager...
                exec(f"from {main_lib}.{sub_lib}.{filename} import *")

# __all__ = []
# from DeepPhysX_Core.AsyncSocket.AbstractEnvironment import AbstractEnvironment
# from DeepPhysX_Core.AsyncSocket.BytesBaseConverter import BytesBaseConverter
# from DeepPhysX_Core.AsyncSocket.BytesNumpyConverter import BytesNumpyConverter
# from DeepPhysX_Core.AsyncSocket.TcpIpClient import TcpIpClient
# from DeepPhysX_Core.AsyncSocket.TcpIpObject import TcpIpObject
# from DeepPhysX_Core.AsyncSocket.TcpIpServer import TcpIpServer
# __all__.append([   "AbstractEnvironment",
#                    "BytesNumpyConverter",
#                    "BytesBaseConverter",
#                    "TcpIpObject",
#                    "TcpIpServer",
#                    "TcpIpClient"])
#
# from DeepPhysX_Core.Dataset.BaseDataset import BaseDataset
# from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
# from DeepPhysX_Core.Dataset.BaseLoader import BaseLoader
# __all__.append(["BaseDataset",
#                 "BaseDatasetConfig",
#                 "BaseLoader"])
#
# from DeepPhysX_Core.Environment.BaseEnvironment import BaseEnvironment
# from DeepPhysX_Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
# __all__.append(["BaseEnvironment",
#                 "BaseEnvironmentConfig"])
#
# from DeepPhysX_Core.Manager.DataManager import DataManager
# from DeepPhysX_Core.Manager.DatasetManager import DatasetManager
# from DeepPhysX_Core.Manager.EnvironmentManager import EnvironmentManager
# from DeepPhysX_Core.Manager.Manager import Manager
# from DeepPhysX_Core.Manager.NetworkManager import NetworkManager
# from DeepPhysX_Core.Manager.StatsManager import StatsManager
# from DeepPhysX_Core.Manager.VisualizerManager import VisualizerManager
# __all__.append(["DataManager",
#            "DatasetManager",
#            "EnvironmentManager",
#            "Manager",
#            "NetworkManager",
#            "StatsManager",
#            "VisualizerManager"])
#
# from DeepPhysX_Core.Network.BaseNetwork import BaseNetwork
# from DeepPhysX_Core.Network.BaseNetworkConfig import BaseNetworkConfig
# from DeepPhysX_Core.Network.BaseOptimization import BaseOptimization
# from DeepPhysX_Core.Network.DataTransformation import DataTransformation
# __all__.append(["BaseNetwork",
#            "BaseNetworkConfig",
#            "BaseOptimization",
#            "DataTransformation"])
#
# from DeepPhysX_Core.Pipelines.BaseDataGenerator import BaseDataGenerator
# from DeepPhysX_Core.Pipelines.BasePipeline import BasePipeline
# from DeepPhysX_Core.Pipelines.BaseRunner import BaseRunner
# from DeepPhysX_Core.Pipelines.BaseTrainer import BaseTrainer
# __all__.append(["BaseDataGenerator",
#            "BasePipeline",
#            "BaseRunner",
#            "BaseTrainer"])
#
# from DeepPhysX_Core.utils.mathUtils import *
# from DeepPhysX_Core.utils.pathUtils import *
# from DeepPhysX_Core.utils.tensor_transform_utils import *
# __all__.append(["nextPowerOf2",
#            "fibonacci3DSphereSampling",
#            "sigmoid",
#            "min_max_feature_scaling",
#            "ndim_interpolation",
#            "createDir",
#            "copyDir",
#            "getFirstCaller",
#            "flatten"])
#
# from DeepPhysX_Core.Visualizer.MeshVisualizer import MeshVisualizer
# from DeepPhysX_Core.Visualizer.SampleVisualizer import SampleVisualizer
# from DeepPhysX_Core.Visualizer.VedoVisualizer import VedoVisualizer
# __all__.append(["MeshVisualizer",
#            "SampleVisualizer",
#            "VedoVisualizer"])
#
# from functools import reduce
# from operator import iconcat
# __all__ = reduce(iconcat, __all__, [])
