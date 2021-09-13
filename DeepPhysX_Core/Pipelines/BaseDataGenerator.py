import os

from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX_Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX_Core.Manager.DataManager import DataManager
import DeepPhysX_Core.utils.pathUtils as pathUtils


class BaseDataGenerator:

    def __init__(self, dataset_config: BaseDatasetConfig, environment_config: BaseEnvironmentConfig,
                 visualizer_class=None, session_name='default', nb_batches=0, batch_size=0):

        session_dir = pathUtils.createDir(os.path.join(pathUtils.getFirstCaller(), session_name),
                                          check_existing=session_name)
        session_name = (session_name if session_name is not None else os.path.basename(session_dir)).split("/")[-1]
        self.data_manager = DataManager(manager=self, dataset_config=dataset_config, environment_config=environment_config,
                                        visualizer_class=visualizer_class, session_name=session_name, session_dir=session_dir,
                                        new_session=True, record_data={'in':True, 'out':True}, batch_size=batch_size)
        self.nb_batch = nb_batches

    def execute(self):
        for i in range(self.nb_batch):
            print("Batch", i)
            self.data_manager.getData()
        self.data_manager.close()
