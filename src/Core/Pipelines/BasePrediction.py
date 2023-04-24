from typing import Optional
from os.path import join, exists

from DeepPhysX.Core.Pipelines.BasePipeline import BasePipeline
from DeepPhysX.Core.Manager.DataManager import DataManager
from DeepPhysX.Core.Manager.NetworkManager import NetworkManager
from DeepPhysX.Core.Network.BaseNetworkConfig import BaseNetworkConfig
from DeepPhysX.Core.Database.BaseDatabaseConfig import BaseDatabaseConfig
from DeepPhysX.Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX.Core.Utils.path import get_first_caller


class BasePrediction(BasePipeline):

    def __init__(self,
                 network_config: BaseNetworkConfig,
                 environment_config: BaseEnvironmentConfig,
                 database_config: Optional[BaseDatabaseConfig] = None,
                 session_dir: str = 'session',
                 session_name: str = 'training',
                 step_nb: int = -1,
                 record: bool = False,
                 network_manager=NetworkManager,
                 data_manager=DataManager):
        """
        BasePrediction is a pipeline defining the running process of an artificial neural network.
        It provides a highly tunable learning process that can be used with any machine learning library.

        :param network_config: Configuration object with the parameters of the Network.
        :param environment_config: Configuration object with the parameters of the Environment.
        :param database_config: Configuration object with the parameters of the Database.
        :param session_dir: Relative path to the directory which contains sessions repositories.
        :param session_name: Name of the new the session repository.
        :param step_nb: Number of simulation step to play.
        :param record: If True, prediction data will be saved in a dedicated Database.
        """

        BasePipeline.__init__(self,
                              network_config=network_config,
                              database_config=database_config,
                              environment_config=environment_config,
                              session_dir=session_dir,
                              session_name=session_name,
                              new_session=False,
                              pipeline='prediction')

        # Define the session repository
        root = get_first_caller()
        session_dir = join(root, session_dir)
        if not exists(join(session_dir, session_name)):
            raise ValueError(f"[{self.name}] The following directory does not exist: {join(session_dir, session_name)}")
        self.session = join(session_dir, session_name)

        # Create a NetworkManager
        self.network_manager = network_manager(network_config=network_config,
                                               pipeline=self.type,
                                               session=self.session,
                                               new_session=False)

        # Create a DataManager
        self.data_manager = data_manager(pipeline=self,
                                         database_config=database_config,
                                         environment_config=environment_config,
                                         session=self.session,
                                         new_session=False,
                                         produce_data=record,
                                         batch_size=1)
        self.data_manager.connect_handler(self.network_manager.get_database_handler())
        self.network_manager.link_clients(self.data_manager.nb_environment)

        # Prediction variables
        self.step_nb = step_nb
        self.step_id = 0

    def execute(self) -> None:
        """
        Launch the prediction Pipeline.
        Each event is already implemented for a basic pipeline but can also be rewritten via inheritance to describe a
        more complex Pipeline.
        """

        self.prediction_begin()
        while self.prediction_condition():
            self.sample_begin()
            self.predict()
            self.sample_end()
        self.prediction_end()

    def prediction_begin(self) -> None:
        """
        Called once at the beginning of the prediction Pipeline.
        """

        pass

    def prediction_condition(self) -> bool:
        """
        Condition that characterize the end of the prediction Pipeline.
        """

        running = self.step_id < self.step_nb if self.step_nb > 0 else True
        self.step_id += 1
        return running

    def sample_begin(self) -> None:
        """
        Called one at the beginning of each sample.
        """

        pass

    def predict(self) -> None:
        """
        Pull the data from the manager and return the prediction in the Exchange table of database handler.
        """

        self.data_manager.get_data(epoch=0,
                                   animate=True)
        if self.data_manager.environment_manager is None:
            # Get the batch from the Database
            batch = self.network_manager.database_handler.get_lines(table_name='Training',
                                                                    lines_id=self.data_manager.data_lines)
            for key in batch.keys():
                if isinstance(batch[key], dict):  # The element is from a referenced table
                    batch[key] = batch[key][key]
                for i, element in enumerate(batch[key]):
                    # Put the batch in the Exchange table
                    self.network_manager.get_database_handler().update(table_name='Exchange',
                                                                       data={key: element},
                                                                       line_id=i+1,
                                                                       create_fields=True)
                    # Compute the prediction from the Exchange table
                    self.network_manager.compute_online_prediction(instance_id=i+1,
                                                                   normalization=self.data_manager.normalization)

    def sample_end(self) -> None:
        """
        Called one at the end of each sample.
        """

        pass

    def prediction_end(self) -> None:
        """
        Called once at the end of the prediction Pipeline.
        """

        self.data_manager.close()
        self.network_manager.close()

    def __str__(self):

        description = BasePipeline.__str__(self)
        description += f"   Number of step: {self.step_nb}\n"
        return description
