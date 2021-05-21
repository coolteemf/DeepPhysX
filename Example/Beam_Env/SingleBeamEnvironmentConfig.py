from .SingleBeamEnvironment import SingleBeamEnvironment
from DeepPhysX_Sofa.Environment.SofaBaseEnvironmentConfig import SofaBaseEnvironmentConfig
from dataclasses import dataclass
import SofaRuntime
import os

class SingleBeamEnvironmentConfig(SofaBaseEnvironmentConfig):
    @dataclass
    class BeamEnvironmentProperties(SofaBaseEnvironmentConfig.SofaBaseEnvironmentProperties):
        p_grid: dict
        p_forces: dict

    def __init__(self, environment_class=SingleBeamEnvironment, simulations_per_step=1, max_wrong_samples_per_step=10,
                 always_create_data=False, multiprocessing=1, multiprocess_method=None, root_node=None,
                 p_grid=None, p_forces=None):
        super(SingleBeamEnvironmentConfig, self).__init__(environment_class, simulations_per_step,
                                                    max_wrong_samples_per_step, always_create_data, multiprocessing,
                                                    multiprocess_method, root_node)
        self.environmentConfig = self.BeamEnvironmentProperties(simulations_per_step=simulations_per_step,
                                                                max_wrong_samples_per_step=max_wrong_samples_per_step,
                                                                p_grid=p_grid, p_forces=p_forces)

    def addRequiredPlugins(self):
        SofaRuntime.PluginRepository.addFirstPath(os.environ['CARIBOU_INSTALL'])
        self.rootNode.addObject('RequiredPlugin', pluginName=['SofaComponentAll', 'SofaLoader', 'SofaCaribou',
                                                              'SofaBaseTopology', 'SofaGeneralEngine', 'SofaEngine',
                                                              'SofaOpenglVisual', 'SofaBoundaryCondition',
                                                              'SofaTopologyMapping'])
