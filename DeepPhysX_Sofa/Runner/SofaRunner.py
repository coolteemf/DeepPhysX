import Sofa

from DeepPhysX_Core.Pipelines.BaseRunner import BaseRunner
from DeepPhysX_Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX_Core.Network.BaseNetworkConfig import BaseNetworkConfig
from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig


class SofaRunner(Sofa.Core.Controller, BaseRunner):

    def __init__(self, network_config: BaseNetworkConfig, dataset_config: BaseDatasetConfig,
                 environment_config: BaseEnvironmentConfig, session_name='default', session_dir=None,
                 nb_steps=0, record_inputs=False, record_outputs=False, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        BaseRunner.__init__(self, network_config=network_config, dataset_config=dataset_config,
                            environment_config=environment_config,
                            session_name=session_name, session_dir=session_dir,
                            nb_steps=nb_steps, record_inputs=record_inputs, record_outputs=record_outputs)
        self.runBegin()
        self.root = self.manager.data_manager.environment_manager.environment.root
        self.root.addObject(self)

    def onAnimateEndEvent(self, event):
        if self.runningCondition():
            self.sampleBegin()
            prediction, loss = self.predict(animate=False)
            self.manager.data_manager.environment_manager.applyPrediction(prediction)
            self.sampleEnd()
        else:
            self.runEnd()
