"""
LiverConfig.py
Configuration class for all the Liver scenes defined in FEMLiver and NNLiver repositories
"""

import os
import SofaRuntime
from dataclasses import dataclass

from DeepPhysX_Sofa.Environment.SofaEnvironmentConfig import SofaEnvironmentConfig


# Inherit from SofaEnvironmentConfig to create Sofa environment in DeepPhysX pipeline
class LiverConfig(SofaEnvironmentConfig):

    def __init__(self, environment_class=None, simulations_per_step=1, always_create_data=False, root_node=None,
                 visualizer_class=None, p_liver=None, p_grid=None, p_force=None):
        # Parent class constructor
        super(LiverConfig, self).__init__(environment_class=environment_class,
                                          simulations_per_step=simulations_per_step,
                                          always_create_data=always_create_data, root_node=root_node,
                                          visualizer_class=visualizer_class)
        # Environment parameters
        self.environment_config = self.LiverProperties(simulations_per_step=simulations_per_step,
                                                       max_wrong_samples_per_step=10,
                                                       p_liver=p_liver, p_grid=p_grid, p_force=p_force)

    @dataclass
    class LiverProperties(SofaEnvironmentConfig.SofaEnvironmentProperties):
        p_liver: dict
        p_grid: dict
        p_force: dict

    def addRequiredPlugins(self):
        SofaRuntime.PluginRepository.addFirstPath(os.environ['CARIBOU_INSTALL'])
        required_plugins = ['SofaComponentAll', 'SofaLoader', 'SofaCaribou', 'SofaBaseTopology', 'SofaGeneralEngine',
                            'SofaEngine', 'SofaOpenglVisual', 'SofaBoundaryCondition']
        self.rootNode.addObject('RequiredPlugin', pluginName=required_plugins)
