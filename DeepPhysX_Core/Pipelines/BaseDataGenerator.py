from os.path import join as osPathJoin
from os.path import basename

from DeepPhysX_Core.Manager.DataManager import DataManager
from DeepPhysX_Core.Utils.pathUtils import createDir, getFirstCaller


class BaseDataGenerator:

    def __init__(self,
                 dataset_config,
                 environment_config,
                 session_name='default',
                 nb_batches=0,
                 batch_size=0,
                 record_input=True,
                 record_output=True):

        session_dir = createDir(osPathJoin(getFirstCaller(), session_name), dir_name=session_name)
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

    def execute(self):
        for i in range(self.nb_batch):
            print("Batch", i)
            self.data_manager.getData()
        self.data_manager.close()
