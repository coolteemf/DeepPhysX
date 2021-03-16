from DeepPhysX.Manager.Manager import Manager


class BaseRunner:

    def __init__(self, session_name, nb_epochs, nb_samples, network_config, dataset_config, environment_config=None,
                 manager_dir=None):

        if environment_config is None and dataset_config.datasetDir is None:
            print("BaseRunner: Need a data source (existing dataset directory or environment). Shutting down.")
            quit(0)

        self.sessionName = session_name
        self.nbEpoch = nb_epochs
        self.nbSamples = nb_samples

        self.datasetConfig = dataset_config
        self.networkConfig = network_config
        self.environmentConfig = environment_config

        self.manager = Manager(session_name=session_name, network_config=network_config, dataset_config=dataset_config,
                               trainer=False, environment_config=environment_config, manager_dir=manager_dir)

    def execute(self):
        """To test network on a large dataset"""
        self.testBegin()
        for epoch in range(self.nbEpoch):
            self.sampleBegin()
            prediction, loss = self.predict()
            self.sampleEnd()
        self.testEnd()

    def predict(self):
        """To do a single prediction"""
        data = self.manager.getData(epoch=0)
        inputs, ground_truth = data['in'], data['out']
        prediction = self.manager.predict(inputs=inputs)
        loss = self.manager.computeLoss(prediction, ground_truth)
        return prediction, loss

    def testBegin(self):
        pass

    def testEnd(self):
        pass

    def epochBegin(self):
        pass

    def epochEnd(self):
        pass

    def sampleBegin(self):
        pass

    def sampleEnd(self):
        pass
