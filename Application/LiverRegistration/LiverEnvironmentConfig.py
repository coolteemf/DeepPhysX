from LiverEnvironment import LiverEnvironment
from DeepPhysX_Sofa.Environment.SofaBaseEnvironmentConfig import SofaBaseEnvironmentConfig, dataclass
import SofaRuntime


class LiverEnvironmentConfig(SofaBaseEnvironmentConfig):
    @dataclass
    class LiverEnvironmentProperties(SofaBaseEnvironmentConfig.SofaBaseEnvironmentProperties):
        p_liver: dict
        p_grid: dict
        p_forces: dict

    def __init__(self, environment_class=LiverEnvironment, simulations_per_step=1, max_wrong_samples_per_step=10,
                 always_create_data=False, multiprocessing=1, multiprocess_method=None, root_node=None,
                 p_liver=None, p_grid=None, p_forces=None):
        super(LiverEnvironmentConfig, self).__init__(environment_class, simulations_per_step,
                                                     max_wrong_samples_per_step,
                                                     always_create_data, multiprocessing, multiprocess_method,
                                                     root_node)

        self.environmentConfig = self.LiverEnvironmentProperties(simulations_per_step=simulations_per_step,
                                                                 max_wrong_samples_per_step=max_wrong_samples_per_step,
                                                                 p_liver=p_liver, p_grid=p_grid, p_forces=p_forces)

    def addRequiredPlugins(self):
        SofaRuntime.PluginRepository.addFirstPath('/home/robin/dev/SofaPython3/build/lib')
        SofaRuntime.PluginRepository.addFirstPath('/home/robin/dev/sofa/build/lib')
        SofaRuntime.PluginRepository.addFirstPath('/home/robin/dev/caribou/build/lib')
        print("Loading Sofa LIBRARIES")
        pluginsNode = self.rootNode.addChild("RequiredPluginsNode")
        pluginsNode.addObject('RequiredPlugin', name='SofaComponentAll')
        pluginsNode.addObject('RequiredPlugin', name='SofaLoader')
        pluginsNode.addObject('RequiredPlugin', name='SofaCaribou')
        pluginsNode.addObject('RequiredPlugin', name='SofaBaseTopology')
        pluginsNode.addObject('RequiredPlugin', name='SofaGeneralEngine')
        pluginsNode.addObject('RequiredPlugin', name='SofaEngine')
        pluginsNode.addObject('RequiredPlugin', name='SofaOpenglVisual')
        pluginsNode.addObject('RequiredPlugin', name='SofaBoundaryCondition')
