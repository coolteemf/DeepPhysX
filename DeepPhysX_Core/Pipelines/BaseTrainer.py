from DeepPhysX_Core.Pipelines.BasePipeline import BasePipeline
from DeepPhysX_Core.Network.BaseNetworkConfig import BaseNetworkConfig
from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX_Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX_Core.Manager.Manager import Manager

from vedo import ProgressBar


class BaseTrainer(BasePipeline):

    def __init__(self, network_config: BaseNetworkConfig, dataset_config: BaseDatasetConfig,
                 environment_config: BaseEnvironmentConfig, session_name='default', session_dir=None,
                 new_session=True, nb_epochs=0, nb_batches=0, batch_size=0):
        """
        BaseTrainer is a pipeline defining the training process of an artificial neural network.
        It provide a highly tunable learning process that can be used with any machine learning library.

        :param network_config: BaseNetworkConfig specialisation containing the parameters of the network manager
        :param dataset_config: BaseDatasetConfig specialisation containing the parameters of the dataset manager
        :param environment_config: BaseEnvironmentConfig specialisation containing the parameters of the environment manager
        :param session_name: String Name of the newly created directory if session_dir is not defined
        :param session_dir: String Name of the directory in which to write all of the neccesary data
        :param new_session: Bool Define the creation of new directories to store data
        :param nb_epochs: int Number of epochs
        :param nb_batches: int Number of batches
        :param batch_size: int Size of a batch
        """

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

        self.epoch_progress_bar = ProgressBar(0, self.nb_epochs, c=1, title="Epochs progress bar")
        self.batch_progress_bar = ProgressBar(0, self.nb_batches, c=2, title="Batches progress bar")

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
        """
        Main function of the training process \"execute\" call the functions associated with the learning process.
        Each of the called functions are already implemented so one can start a basic training.
        Each of the called function can also be rewritten via inheritance to provide more specific / complex training process.
        :return:
        """
        self.trainBegin()
        while self.epochCondition():
            self.epochBegin()
            while self.batchCondition():
                self.batchBegin()
                self.optimize()
                self.batchCount()
                self.batchEnd()
            self.epochCount()
            self.epochEnd()
            self.saveNetwork()
        self.trainEnd()

    def optimize(self):
        """
        Pulls data from the manager and run a prediction and optimizer step.
        :return:
        """
        self.manager.getData(self.id_epoch, self.batch_size)
        self.loss_value = self.manager.optimizeNetwork()

    def saveNetwork(self):
        """
        Registers the network weights and biases in the corresponding directory (session_name/network or session_dir/network)
        :return:
        """
        self.manager.saveNetwork()

    def trainBegin(self):
        """
        Called once at the very beginning of the training process.
        Allows the user to run some pre-computations.
        :return:
        """
        pass

    def trainEnd(self):
        """
        Called once at the very end of the training process.
        Allows the user to run some post-computations.
        :return:
        """
        self.manager.close()

    def epochBegin(self):
        """
        Called one at the start of each epoch.
        Allows the user to run some pre-epoch computations.
        :return:
        """
        self.id_batch = 0
        self.manager.data_manager.dataset_manager.dataset.shuffle()

    def epochEnd(self):
        """
        Called one at the end of each epoch.
        Allows the user to run some post-epoch computations.
        :return:
        """
        self.batch_progress_bar.print(counts=self.id_batch)
        self.epoch_progress_bar.print(counts=self.id_epoch)
        self.manager.statsManager.add_trainEpochLoss(self.loss['item'], self.epoch)
        pass

    def epochCondition(self):
        """
        Condition that characterize the end of the training process
        :return: Boolean : False if the training needs to stop.
        """
        return self.id_epoch < self.nb_epochs

    def epochCount(self):
        """
        Allows user for custom update of epochs count
        :return:
        """
        self.id_epoch += 1

    def batchBegin(self):
        """
        Called one at the start of each batch.
        Allows the user to run some pre-batch computations.
        :return:
        """
        pass

    def batchEnd(self):
        """
        Called one at the start of each batch.
        Allows the user to run some post-batch computations.
        :return:
        """
        self.batch_progress_bar.print(counts=self.id_batch)
        self.epoch_progress_bar.print(counts=self.id_epoch)
        self.manager.stats_manager.add_trainBatchLoss(self.loss_value, self.id_epoch * self.nb_batches + self.id_batch)
        pass

    def batchCondition(self):
        """
        Condition that characterize the end of the epoch
        :return: Boolean : False if the epoch needs to stop.
        """
        return self.id_batch < self.nb_batches


    def batchCount(self):
        """
        Allows user for custom update of batches count
        :return:
        """
        self.id_batch += 1

    # def validate(self, size):
    #     success_count = 0
    #     with no_grad():
    #         for i in range(size):
    #             inputs, ground_truth = self.manager.environment_manager.getData(1, True, True)
    #             prediction = np.argmax(self.manager.getPrediction(inputs=inputs).numpy())
    #             if prediction == ground_truth[0]:
    #                 success_count += 1
    #     print("Success Rate =", success_count * 100.0 / size)