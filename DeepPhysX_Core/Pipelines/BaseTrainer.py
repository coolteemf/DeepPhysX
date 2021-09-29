from DeepPhysX_Core.Pipelines.BasePipeline import BasePipeline
from DeepPhysX_Core.Manager.Manager import Manager

from vedo import ProgressBar
from sys import stdout

class BaseTrainer(BasePipeline):

    def __init__(self, network_config,
                 dataset_config,
                 environment_config,
                 session_name='default',
                 session_dir=None,
                 new_session=True,
                 nb_epochs=0, nb_batches=0, batch_size=0):
        """
        BaseTrainer is a pipeline defining the training process of an artificial neural network.
        It provide a highly tunable learning process that can be used with any machine learning library.

        :param BaseNetworkConfig network_config: Specialisation containing the parameters of the network manager
        :param BaseDatasetConfig dataset_config: Specialisation containing the parameters of the dataset manager
        :param BaseEnvironmentConfig environment_config: Specialisation containing the parameters of the environment manager
        :param str session_name: Name of the newly created directory if session_dir is not defined
        :param str session_dir: Name of the directory in which to write all of the neccesary data
        :param bool new_session: Define the creation of new directories to store data
        :param int nb_epochs: Number of epochs
        :param int nb_batches: Number of batches
        :param int batch_size: Size of a batch
        """

        self.name = self.__class__.__name__

        if environment_config is None and dataset_config.dataset_dir is None:
            print("BaseTrainer: You have to give me a dataset source (existing dataset directory or simulation to "
                  "create data on the fly")
            quit(0)

        BasePipeline.__init__(self, network_config=network_config, dataset_config=dataset_config,
                              environment_config=environment_config,
                              session_name=session_name, session_dir=session_dir, pipeline='training')

        self.manager = Manager(pipeline=self, network_config=self.network_config, dataset_config=dataset_config,
                               environment_config=self.environment_config,
                               session_name=session_name,
                               session_dir=session_dir, new_session=new_session, batch_size=batch_size)
        # Training variables
        self.nb_epochs = nb_epochs
        self.id_epoch = 0
        self.nb_batches = nb_batches
        self.batch_size = batch_size
        self.id_batch = 0
        self.nb_samples = nb_batches * batch_size * nb_epochs
        self.loss_dict = None

        # Tell if data is recording while predicting (output is recorded only if input too)
        self.record_data = {"in": True, "out": True}

        self.training_progress_bar = ProgressBar(start=0, stop=self.nb_samples, c='orange', title="Training")
        self.manager.saveInfoFile()

        self.manager.saveInfoFile()

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
        prediction, self.loss_dict = self.manager.optimizeNetwork()
        print(f"Current loss : {self.loss_dict['loss']}")
        self.manager.data_manager.applyPrediction(prediction)

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
        stdout.write("\033[K")
        self.training_progress_bar.print(counts=self.nb_samples)
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
        self.manager.stats_manager.add_trainEpochLoss(self.loss_dict['loss'], self.id_epoch)
        pass

    def epochCondition(self):
        """
        Condition that characterize the end of the training process
        
        :return: bool : False if the training needs to stop.
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
        print(f'Epoch n°{self.id_epoch + 1}/{self.nb_epochs} - Batch n°{self.id_batch + 1}/{self.nb_batches}')
        pass

    def batchEnd(self):
        """
        Called one at the start of each batch.
        Allows the user to run some post-batch computations.
        
        :return:
        """
        stdout.write("\033[K")
        self.training_progress_bar.print(txt=f'Epoch n°{self.id_epoch + 1}/{self.nb_epochs} - ' +
                                             f'Batch n°{self.id_batch + 1}/{self.nb_batches}',
                                         counts=self.nb_batches * self.id_epoch + self.id_batch)
        self.manager.stats_manager.add_trainBatchLoss(self.loss_dict['loss'],
                                                      self.id_epoch * self.nb_batches + self.id_batch)
        for key in self.loss_dict.keys():
            if key != 'loss':
                self.manager.stats_manager.add_customScalar(tag=key, value=self.loss_dict[key],
                                                            count=self.id_epoch * self.nb_batches + self.id_batch)

    def batchCondition(self):
        """
        Condition that characterize the end of the epoch
        
        :return: bool : False if the epoch needs to stop.
        """
        return self.id_batch < self.nb_batches

    def batchCount(self):
        """
        Allows user for custom update of batches count
        
        :return:
        """
        self.id_batch += 1

    def __str__(self):
        """
        
        :return: str Contains training informations about the training process
        """
        description = "\n"
        description += f"# {self.name}\n"
        description += f"    Session directory: {self.manager.session_dir}\n"
        description += f"    Number of epochs: {self.nb_epochs}\n"
        description += f"    Number of batches per epoch: {self.nb_batches}\n"
        description += f"    Number of samples per epoch: {self.nb_samples}\n"
        description += f"    Number of samples per batch: {self.batch_size}\n"
        description += f"    Total: Number of batches : {self.nb_batches * self.nb_epochs}\n"
        description += f"           Number of samples : {self.nb_epochs * self.nb_samples}\n"
        return description

# def validate(self, size):
    #     success_count = 0
    #     with no_grad():
    #         for i in range(size):
    #             inputs, ground_truth = self.manager.environment_manager.getData(1, True, True)
    #             prediction = np.argmax(self.manager.getPrediction(inputs=inputs).numpy())
    #             if prediction == ground_truth[0]:
    #                 success_count += 1
    #     print("Success Rate =", success_count * 100.0 / size)
