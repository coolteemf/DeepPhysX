from DeepPhysX.Manager.Manager import Manager


class BaseRunner:

    def __init__(self, session_name, network_config, dataset_config, environment_config=None,
                 manager_dir=None, nb_samples=0):

        if environment_config is None and dataset_config.datasetDir is None:
            print("BaseRunner: Need a data source (existing dataset directory or environment). Shutting down.")
            quit(0)

        self.sessionName = session_name
        self.nbSamples = nb_samples
        self.idxSample = 0

        self.datasetConfig = dataset_config
        self.networkConfig = network_config
        self.environmentConfig = environment_config

        self.manager = Manager(session_name=session_name, network_config=network_config, dataset_config=dataset_config,
                               trainer=False, environment_config=environment_config, manager_dir=manager_dir)

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
        running = self.idxSample < self.nbSamples if self.nbSamples > 0 else True
        self.idxSample += 1
        return running

    def sampleBegin(self):
        pass

    def sampleEnd(self):
        pass
