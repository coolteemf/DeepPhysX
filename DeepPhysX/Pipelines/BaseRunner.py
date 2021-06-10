from DeepPhysX.Pipelines.BasePipeline import BasePipeline
from DeepPhysX.Manager.Manager import Manager
from DeepPhysX.Network.BaseNetworkConfig import BaseNetworkConfig
from DeepPhysX.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig


class BaseRunner(BasePipeline):

    def __init__(self, network_config: BaseNetworkConfig, dataset_config: BaseDatasetConfig,
                 environment_config: BaseEnvironmentConfig, session_name='default', session_dir=None,
                 nb_steps=0, record_inputs=False, record_outputs=False):

        BasePipeline.__init__(self, pipeline='prediction')

        self.name = self.__class__.__name__

        # Todo: check arguments
        # Check the arguments
        if not isinstance(network_config, BaseNetworkConfig):
            raise TypeError("[BaseRunner] The network configuration must be a BaseNetworkConfig")
        if not isinstance(environment_config, BaseEnvironmentConfig):
            raise TypeError("[BaseRunner] The environment configuration must be a BaseEnvironmentConfig")
        if dataset_config is not None and not isinstance(dataset_config, BaseDatasetConfig):
            raise TypeError("[BaseRunner] The dataset configuration must be a BaseDatasetConfig")
        if type(session_name) != str:
            raise TypeError("[BaseRunner] The network config must be a BaseNetworkConfig object.")
        if session_dir is not None and type(session_dir) != str:
            raise TypeError("[BaseRunner] The session directory must be a str.")
        if type(nb_steps) != int or nb_steps < 0:
            raise TypeError("[BaseRunner] The number of steps must be a positive int")

        self.nb_samples = nb_steps
        self.idx_step = 0
        self.prediction = None
        self.loss_value = 0.

        # Tell if data is recording while predicting (output is recorded only if input too)
        self.record_data = {'in': record_inputs,
                            'out': record_outputs and record_inputs}

        self.manager = Manager(pipeline=self, network_config=network_config, dataset_config=dataset_config,
                               environment_config=environment_config, session_name=session_name,
                               session_dir=session_dir)

    def execute(self):
        self.runBegin()
        while self.runningCondition():
            self.sampleBegin()
            self.prediction, self.loss_value = self.predict()
            self.sampleEnd()
        self.runEnd()

    def predict(self, animate=True):
        self.manager.getData(animate=animate)
        return self.manager.getPrediction()

    def runBegin(self):
        pass

    def runEnd(self):
        pass

    def runningCondition(self):
        running = self.idx_step < self.nb_samples if self.nb_samples > 0 else True
        self.idx_step += 1
        return running

    def sampleBegin(self):
        pass

    def sampleEnd(self):
        pass

    def close(self):
        self.manager.close()
