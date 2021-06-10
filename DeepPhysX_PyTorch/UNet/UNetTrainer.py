from DeepPhysX.Pipelines.BaseTrainer import BaseTrainer


class UNet_Trainer(BaseTrainer):

    def __init__(self, session_name, nb_epochs, nb_batches, batch_size,
                 network_config, dataset_config, environment_config=None, session_dir=None):
        super(UNet_Trainer, self).__init__(session_name, nb_epochs, nb_batches, batch_size,
                                           network_config, dataset_config, environment_config, session_dir)
