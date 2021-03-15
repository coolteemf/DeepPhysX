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
        self.batchSize = batch_size
        self.nbBatches = nb_batches
        self.nbSamples = nb_batches * batch_size

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
        for epoch in range(self.nbEpochs):
            self.epochBegin()
            for batch in range(self.nbBatches):
                self.batchBegin()
                data = self.manager.getData(epoch=epoch, batch_size=self.batchSize)
                inputs, ground_truth = data['in'], data['out']
                prediction = self.manager.predict(inputs=inputs)
                loss = self.manager.optimizeNetwork(prediction=prediction, ground_truth=ground_truth)
                print(loss.item())
                self.batchEnd()
            self.epochEnd()
        self.trainEnd()

    def validate(self, size):
        success_count = 0
        # TODO: find alternative to no_grad()
        # TODO: move from this file
        with no_grad():
            for i in range(size):
                data = self.manager.environmentManager.getData(1, True, True)
                inputs, ground_truth = data['in'], data['out']
                prediction = np.argmax(self.manager.predict(inputs=inputs).numpy())
                if prediction == ground_truth[0]:
                    success_count += 1
        print("Success Rate =", success_count * 100.0 / size)

    def trainBegin(self):
        pass

    def trainEnd(self):
        #self.manager.close()
        pass

    def epochBegin(self):
        self.manager.datasetManager.dataset.shuffle()

    def epochEnd(self):
        pass

    def batchBegin(self):
        pass

    def batchEnd(self):
        pass
