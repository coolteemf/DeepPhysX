from DeepPhysX.Runner.BaseRunner import BaseRunner


class SofaBaseRunner(BaseRunner):

    def __init__(self, session_name, network_config, dataset_config, environment_config=None,
                 manager_dir=None, nb_samples=0):

        super().__init__(session_name, network_config, dataset_config, environment_config, manager_dir,
                         nb_samples)
        self.runBegin()

    def execute(self):
        self.sampleBegin()
        prediction, loss = self.predict()
        self.sampleEnd()

    def onAnimateEndEvent(self, event):
        if self.runningCondition():
            self.execute()
        else:
            self.runEnd()
