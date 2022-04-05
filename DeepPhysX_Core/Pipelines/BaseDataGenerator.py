from os.path import join as osPathJoin
from os.path import basename
from sys import stdout

from DeepPhysX_Core.Manager.DataManager import DataManager
from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX_Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX_Core.Utils.progressbar import Progressbar
from DeepPhysX_Core.Utils.pathUtils import create_dir, get_first_caller


class BaseDataGenerator:
    """
    | BaseDataGenerator implement a minimalist execute function that simply produce and store data without
      training a neural network.

    :param BaseDatasetConfig dataset_config: Specialisation containing the parameters of the dataset manager
    :param BaseEnvironmentConfig environment_config: Specialisation containing the parameters of the environment manager
    :param str session_name: Name of the newly created directory if session_dir is not defined
    :param int nb_batches: Number of batches
    :param int batch_size: Size of a batch
    :param bool record_input: True if the input must be stored
    :param bool record_output: True if the output must be stored
    """

    def __init__(self,
                 dataset_config: BaseDatasetConfig,
                 environment_config: BaseEnvironmentConfig,
                 session_name: str = 'default',
                 session_dir: str = None,
                 nb_batches: int = 0,
                 batch_size: int = 0,
                 record_input: bool = True,
                 record_output: bool = True):

        # todo: inherit from Pipeline
        # Init session repository
        if session_dir is None:
            # Create manager directory from the session name
            self.session_dir: str = create_dir(osPathJoin(get_first_caller(), session_name), dir_name=session_name)
        else:
            self.session_dir: str = osPathJoin(session_dir, session_name)
        session_name = basename(self.session_dir).split("/")[-1]

        # Create a DataManager directly
        self.data_manager = DataManager(manager=self,
                                        dataset_config=dataset_config,
                                        environment_config=environment_config,
                                        session_name=session_name,
                                        session_dir=self.session_dir,
                                        new_session=True,
                                        record_data={'input': record_input, 'output': record_output},
                                        batch_size=batch_size)
        self.nb_batch: int = nb_batches
        self.progress_bar = Progressbar(start=0, stop=self.nb_batch, c='orange', title="Data Generation")

    def execute(self) -> None:
        """
        | Run the data generation and recording process.
        """

        for i in range(self.nb_batch):
            # Update progress bar
            stdout.write("\033[K")
            self.progress_bar.print(counts=i + 1)
            # Produce a batch
            self.data_manager.get_data()
        # Close manager
        self.data_manager.close()
