from os.path import join as osPathJoin
from os.path import basename

from DeepPhysX_Core.Manager.DataManager import DataManager
from DeepPhysX_Core.Utils.pathUtils import create_dir, get_first_caller

from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX_Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig


class BaseDataGenerator:
    """
    Gives minimal process to generate a dataset
        Description:
            BaseDataGenerator implement a minimalist execute function that simply produce and store data without
            training a neural network.
    """

    data_manager: DataManager
    nb_batch: int

    def __init__(self,
                 dataset_config,
                 environment_config,
                 session_name='default',
                 nb_batches=0,
                 batch_size=0,
                 record_input=True,
                 record_output=True):
        """
        Initialize a minimalist class that simply produce and store data without training a neural network.

        :param BaseDatasetConfig dataset_config: Specialisation containing the parameters of the dataset manager
        :param BaseEnvironmentConfig environment_config: Specialisation containing the parameters of the environment manager
        :param str session_name: Name of the newly created directory if session_dir is not defined
        :param int nb_batches: Number of batches
        :param int batch_size: Size of a batch
        :param record_input: True if the input must be stored
        :param record_output: True if the output must be stored
        """

        session_dir = create_dir(osPathJoin(get_first_caller(), session_name), dir_name=session_name)
        session_name = (session_name if session_name is not None else basename(session_dir)).split("/")[-1]
        self.data_manager = DataManager(manager=self,
                                        dataset_config=dataset_config,
                                        environment_config=environment_config,
                                        session_name=session_name,
                                        session_dir=session_dir,
                                        new_session=True,
                                        record_data={'input': record_input, 'output': record_output},
                                        batch_size=batch_size)
        self.nb_batch = nb_batches

    def execute(self) -> None:
        """Run the data generation and recording process"""
        for i in range(self.nb_batch):
            print("Batch", i)
            self.data_manager.getData()
        self.data_manager.close()
