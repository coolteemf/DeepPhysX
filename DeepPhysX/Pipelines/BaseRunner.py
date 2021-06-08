from DeepPhysX.Pipelines.BasePipeline import BasePipeline
from DeepPhysX.Manager.Manager import Manager
from DeepPhysX.Network.BaseNetworkConfig import BaseNetworkConfig
from DeepPhysX.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig


class BaseRunner(BasePipeline):

    def __init__(self, network_config=BaseNetworkConfig(), dataset_config=BaseDatasetConfig(),
                 environment_config=BaseEnvironmentConfig(), session_name='default', session_dir=None,
                 nb_samples=0, record_inputs=False, record_outputs=False):

        BasePipeline.__init__(self, pipeline='prediction')

        # Todo: check arguments
        # Check the arguments
        if not isinstance(network_config, BaseNetworkConfig):
            raise TypeError("[BASERUNNER] The network configuration must be a BaseNetworkConfig")
        if not isinstance(environment_config, BaseEnvironmentConfig):
            raise TypeError("[BASERUNNER] The environment configuration must be a BaseEnvironmentConfig")
        if dataset_config is not None and not isinstance(dataset_config, BaseDatasetConfig):
            raise TypeError("[BASERUNNER] The dataset configuration must be a BaseDatasetConfig")

        self.nb_samples = nb_samples
        self.idx_sample = 0

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
            prediction, loss = self.predict()
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
        running = self.idx_sample < self.nb_samples if self.nb_samples > 0 else True
        self.idx_sample += 1
        return running

    def sampleBegin(self):
        pass

    def sampleEnd(self):
        pass
