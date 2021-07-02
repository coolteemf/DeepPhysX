"""
BeamConfig.py
Configuration class for all the Beam scenes defined in FEMBeam and NNBeam repositories
"""

import os
import SofaRuntime
from dataclasses import dataclass

from DeepPhysX_Sofa.Environment.SofaEnvironmentConfig import SofaEnvironmentConfig


# Inherit from SofaEnvironmentConfig to create Sofa environment in DeppPhysX pipeline
class BeamConfig(SofaEnvironmentConfig):

    def __init__(self, environment_class=None, simulations_per_step=1, always_create_data=False, root_node=None,
                 visualizer_class=None, p_grid=None):
        # Parent class constructor
        super(BeamConfig, self).__init__(environment_class=environment_class, simulations_per_step=simulations_per_step,
                                         always_create_data=always_create_data, root_node=root_node,
                                         visualizer_class=visualizer_class)
        # Environment parameters
        self.environment_config = self.BeamProperties(simulations_per_step=simulations_per_step,
                                                      max_wrong_samples_per_step=10,
                                                      p_grid=p_grid)

    @dataclass
    # Inherit from SofaEnvironmentProperties to add custom parameters in the scene configuration
    class BeamProperties(SofaEnvironmentConfig.SofaEnvironmentProperties):
        p_grid: dict  # Define the beam geometry (size, resolution...)

    def addRequiredPlugins(self):
        # Todo: avoid using personal environment variable (or add instructions in a readme)
        SofaRuntime.PluginRepository.addFirstPath(os.environ['CARIBOU_INSTALL'])
        required_plugins = ['SofaComponentAll', 'SofaLoader', 'SofaCaribou', 'SofaBaseTopology', 'SofaGeneralEngine',
                            'SofaEngine', 'SofaOpenglVisual', 'SofaBoundaryCondition', 'SofaTopologyMapping',
                            'SofaConstraint', 'SofaDeformable', 'SofaGeneralObjectInteraction', 'SofaBaseMechanics',
                            'SofaMiscCollision']
        self.rootNode.addObject('RequiredPlugin', pluginName=required_plugins)
