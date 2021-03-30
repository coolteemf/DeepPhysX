import numpy as np
from torch import no_grad

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
        self.epoch = 0
        self.nbBatches = nb_batches
        self.batchSize = batch_size
        self.batch = 0
        self.nbSamples = nb_batches * batch_size
        self.loss = None

        # Dataset variables
        self.datasetConfig = dataset_config

        # Network variables
        self.networkConfig = network_config

        # Simulation variables
        self.environmentConfig = environment_config

        self.manager = Manager(session_name=self.sessionName, network_config=self.networkConfig,
                               dataset_config=dataset_config, trainer=True, environment_config=self.environmentConfig,
                               manager_dir=manager_dir)

    def execute(self):
        self.trainBegin()
        while self.epochCondition():
            self.epochBegin()
            self.batch = 0
            while self.batchCondition():
                self.batchBegin()
                self.loss = self.manager.optimizeNetwork(self.epoch, self.batchSize)
                self.batchTrainingIndicator()
                self.incrementBatch()
                self.batchEnd()
            self.epochTrainingIndicator()
            self.incrementEpoch()
            self.epochEnd()
            self.saveNetwork()
        self.trainEnd()

    def validate(self, size):
        success_count = 0
        # TODO: find alternative to no_grad()
        # TODO: move from this file
        with no_grad():
            for i in range(size):
                inputs, ground_truth = self.manager.environmentManager.getData(1, True, True)
                prediction = np.argmax(self.manager.getPrediction(inputs=inputs).numpy())
                if prediction == ground_truth[0]:
                    success_count += 1
        print("Success Rate =", success_count * 100.0 / size)

    def trainBegin(self):
        pass

    def trainEnd(self):
        self.manager.close()

    def epochBegin(self):
        self.manager.datasetManager.dataset.shuffle()

    def epochCondition(self):
        return self.epoch < self.nbEpochs

    def epochTrainingIndicator(self):
        self.manager.statsManager.add_trainEpochLoss(self.loss.item(), self.epoch)
        if self.epoch % 100 == 0:
            print(self.loss.item())

    def incrementEpoch(self):
        self.epoch += 1

    def epochEnd(self):
        pass

    def saveNetwork(self):
        self.manager.saveNetwork()

    def batchBegin(self):
        pass

    def batchCondition(self):
        return self.batch < self.nbBatches

    def batchTrainingIndicator(self):
        self.manager.statsManager.add_trainBatchLoss(self.loss.item(), self.epoch * self.nbBatches + self.batch)
        self.manager.statsManager.add_trainTestBatchLoss(self.loss.item(), None, self.epoch * self.nbBatches + self.batch)  # why ?

    def incrementBatch(self):
        self.batch += 1

    def batchEnd(self):
        pass
