from DeepPhysX_Core.Pipelines.BasePipeline import BasePipeline
from DeepPhysX_Core.Manager.Manager import Manager

from DeepPhysX_Core.Network.BaseNetworkConfig import BaseNetworkConfig
from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX_Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig


class BaseTrainer(BasePipeline):

    def __init__(self,
                 network_config,
                 dataset_config,
                 environment_config,
                 session_name='default',
                 session_dir=None,
                 new_session=True,
                 nb_epochs=0,
                 nb_batches=0,
                 batch_size=0):
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
        self.record_data = {'input': True, 'output': True}

        self.manager.save_info_file()

    def execute(self) -> None:
        """
        Main function of the training process \"execute\" call the functions associated with the learning process.
        Each of the called functions are already implemented so one can start a basic training.
        Each of the called function can also be rewritten via inheritance to provide more specific / complex training process.
        """
        self.train_begin()
        while self.epoch_condition():
            self.epoch_begin()
            while self.batch_condition():
                self.batch_begin()
                self.optimize()
                self.batch_count()
                self.batch_end()
            self.epoch_count()
            self.epoch_end()
            self.save_network()
        self.train_end()

    def optimize(self) -> None:
        """
        Pulls data from the manager and run a prediction and optimizer step.

        """
        self.manager.get_data(self.id_epoch, self.batch_size)
        prediction, self.loss_dict = self.manager.optimize_network()
        print(f"Current loss : {self.loss_dict['loss']}")

    def save_network(self) -> None:
        """
        Registers the network weights and biases in the corresponding directory (session_name/network or session_dir/network)
        """
        self.manager.save_network()

    def train_begin(self) -> None:
        """
        Called once at the very beginning of the training process.
        Allows the user to run some pre-computations.
        """
        pass

    def train_end(self) -> None:
        """
        Called once at the very end of the training process.
        Allows the user to run some post-computations.
        """
        self.manager.close()

    def epoch_begin(self) -> None:
        """
        Called one at the start of each epoch.
        Allows the user to run some pre-epoch computations.
        """
        self.id_batch = 0

    def epoch_end(self) -> None:
        """
        Called one at the end of each epoch.
        Allows the user to run some post-epoch computations.
        """
        self.manager.stats_manager.add_train_epoch_loss(self.loss_dict['loss'], self.id_epoch)

    def epoch_condition(self) -> bool:
        """
        Condition that characterize the end of the training process
        
        :return: bool : False if the training needs to stop.
        """
        return self.id_epoch < self.nb_epochs

    def epoch_count(self) -> None:
        """
        Allows user for custom update of epochs count
        """
        self.id_epoch += 1

    def batch_begin(self) -> None:
        """
        Called one at the start of each batch.
        Allows the user to run some pre-batch computations.
        """
        print(f'Epoch n°{self.id_epoch + 1}/{self.nb_epochs} - Batch n°{self.id_batch + 1}/{self.nb_batches}')

    def batch_end(self) -> None:
        """
        Called one at the start of each batch.
        Allows the user to run some post-batch computations.
        """
        self.manager.stats_manager.add_train_batch_loss(self.loss_dict['loss'],
                                                      self.id_epoch * self.nb_batches + self.id_batch)
        for key in self.loss_dict.keys():
            if key != 'loss':
                self.manager.stats_manager.add_custom_scalar(tag=key, value=self.loss_dict[key],
                                                            count=self.id_epoch * self.nb_batches + self.id_batch)

    def batch_condition(self) -> bool:
        """
        Condition that characterize the end of the epoch
        
        :return: bool : False if the epoch needs to stop.
        """
        return self.id_batch < self.nb_batches

    def batch_count(self):
        """
        Allows user for custom update of batches count
        
        :return:
        """
        self.id_batch += 1

    def __str__(self) -> str:
        """
        
        :return: str Contains training informations about the training process
        """
        description = "\n"
        description += f"# {self.__class__.__name__}\n"
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
