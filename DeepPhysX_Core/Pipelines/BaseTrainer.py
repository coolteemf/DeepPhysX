from DeepPhysX_Core.Pipelines.BasePipeline import BasePipeline
from DeepPhysX_Core.Network.BaseNetworkConfig import BaseNetworkConfig
from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX_Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX_Core.Manager.Manager import Manager


class BaseTrainer(BasePipeline):

    def __init__(self, network_config: BaseNetworkConfig, dataset_config: BaseDatasetConfig,
                 environment_config: BaseEnvironmentConfig, session_name='default', session_dir=None,
                 new_session=True, nb_epochs=0, nb_batches=0, batch_size=0):

        BasePipeline.__init__(self, pipeline='training')

        if environment_config is None and dataset_config.dataset_dir is None:
            print("BaseTrainer: You have to give me a dataset source (existing dataset directory or simulation to "
                  "create data on the fly")
            quit(0)

        # Storage variables

        # Training variables
        self.nb_epochs = nb_epochs
        self.id_epoch = 0
        self.nb_batches = nb_batches
        self.batch_size = batch_size
        self.id_batch = 0
        self.nb_samples = nb_batches * batch_size
        self.loss_value = None

        # Testing variables

        # Dataset variables
        self.dataset_config = dataset_config

        # Network variables
        self.network_config = network_config

        # Simulation variables
        self.environment_config = environment_config

        self.manager = Manager(pipeline=self, network_config=self.network_config, dataset_config=dataset_config,
                               environment_config=self.environment_config, session_name=session_name,
                               session_dir=session_dir, new_session=new_session)

    def execute(self):
        self.trainBegin()
        while self.epochCondition():
            self.epochBegin()
            while self.batchCondition():
                self.batchBegin()
                self.optimize()
                self.batchStats()
                self.batchCount()
                self.batchEnd()
            self.epochStats()
            self.epochCount()
            self.epochEnd()
            self.saveNetwork()
        self.trainEnd()

    def optimize(self):
        self.manager.getData(self.id_epoch, self.batch_size)
        self.loss_value = self.manager.optimizeNetwork()

    def saveNetwork(self):
        self.manager.saveNetwork()

    def trainBegin(self):
        pass

    def trainEnd(self):
        self.manager.close()

    def epochBegin(self):
        self.id_batch = 0
        self.manager.data_manager.dataset_manager.dataset.shuffle()
        print("Epoch", self.id_epoch)

    def epochEnd(self):
        pass

    def epochCondition(self):
        return self.id_epoch < self.nb_epochs

    def epochStats(self):
        # self.manager.statsManager.add_trainEpochLoss(self.loss['item'], self.epoch)
        if (self.id_epoch + 1) % 5 == 0:
            print(self.loss_value)

    def epochCount(self):
        self.id_epoch += 1

    def batchBegin(self):
        pass

    def batchEnd(self):
        pass

    def batchCondition(self):
        return self.id_batch < self.nb_batches

    def batchStats(self):
        self.manager.stats_manager.add_trainBatchLoss(self.loss_value, self.id_epoch * self.nb_batches + self.id_batch)
        # self.manager.statsManager.add_trainTestBatchLoss(self.loss['item'], None, self.epoch * self.nbBatches + self.batch)  # why ?

    def batchCount(self):
        self.id_batch += 1
        print(self.id_batch, self.loss_value)

    # def validate(self, size):
    #     success_count = 0
    #     with no_grad():
    #         for i in range(size):
    #             inputs, ground_truth = self.manager.environment_manager.getData(1, True, True)
    #             prediction = np.argmax(self.manager.getPrediction(inputs=inputs).numpy())
    #             if prediction == ground_truth[0]:
    #                 success_count += 1
    #     print("Success Rate =", success_count * 100.0 / size)
