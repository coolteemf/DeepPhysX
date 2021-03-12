from DeepPhysX.Manager.Manager import Manager


class BaseTrainer:

    def __init__(self, session_name, nb_epochs, nb_batches, batch_size,
                 network_config, dataset_config, environment_config=None, manager_dir=None):

        if environment_config is None and dataset_config.datasetDir is None:
            print("BaseTrainer: You have to give me a dataset source (existing dataset directory or simulation to "
                  "create data on the fly")
            quit(0)

        # Training variables
        self.sessionName = session_name
        self.nbEpochs = nb_epochs
        self.batchSize = batch_size
        self.nbBatches = nb_batches
        self.nbSamples = nb_batches * batch_size

        # Dataset variables
        self.datasetConfig = dataset_config

        # Network variables
        self.networkConfig = network_config

        # Simulation variables
        self.environmentConfig = environment_config
        # Todo: EnvironmentConfig
        self.alwaysCreateData = environment_config.alwaysCreateData

        self.manager = Manager(session_name=self.sessionName, network_config=self.networkConfig,
                               dataset_config=dataset_config, trainer=True, environment_config=self.environmentConfig,
                               manager_dir=manager_dir)

    def execute(self):
        self.trainBegin()
        for epoch in range(self.nbEpochs):
            self.epochBegin()
            for batch in range(self.nbBatches):
                self.batchBegin()
                inputs, expects = self.getData(epoch)
                predicts = self.manager.networkManager.network.predict(inputs)
                loss = self.manager.networkManager.updateNetwork(predicts, expects)
                self.batchEnd()
            self.epochEnd()
        self.trainEnd()

    def validate(self):
        pass

    def trainBegin(self):
        pass

    def trainEnd(self):
        pass

    def epochBegin(self):
        pass

    def epochEnd(self):
        pass

    def batchBegin(self):
        pass

    def batchEnd(self):
        pass

    def getData(self, epoch):
        if (self.environmentConfig is not None) and (epoch == 1 or self.alwaysCreateData):
            return self.manager.getData(source='environment', batch_size=self.batchSize,
                                        get_inputs=True, get_outputs=True)
        else:
            return self.manager.getData(source='dataset', batch_size=self.batchSize,
                                        get_inputs=True, get_outputs=True)
