import sys
import torch

from DeepPhysX_Sofa.Environment.SofaEnvironmentConfig import SofaEnvironmentConfig, BytesNumpyConverter
from DeepPhysX_PyTorch.FC.FCConfig import FCConfig
from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX_Core.Pipelines.BaseTrainer import BaseTrainer
from Tests.AsyncSocket.EnvironmentSofa import EnvironmentSofa

if len(sys.argv) != 2:
    print(f"Usage: python3 {sys.argv[0]} <nb_thread>")
    sys.exit(1)

env_config = SofaEnvironmentConfig(environment_class=EnvironmentSofa,
                                   environment_file=sys.modules[EnvironmentSofa.__module__].__file__,
                                   number_of_thread=int(sys.argv[1]),
                                   socket_data_converter=BytesNumpyConverter,
                                   always_create_data=True)

neurones_per_layer = 50*50*3  # Small RGB image of size 50x50
network_config = FCConfig(loss=torch.nn.MSELoss, lr=1e-3, optimizer=torch.optim.Adam,
                          dim_layers=[neurones_per_layer, neurones_per_layer, neurones_per_layer], dim_output=3)

dataset_config = BaseDatasetConfig(partition_size=1, shuffle_dataset=True)

trainer = BaseTrainer(session_name='testing/test', dataset_config=dataset_config, environment_config=env_config,
                      network_config=network_config, nb_epochs=5, nb_batches=10, batch_size=5)
trainer.execute()
