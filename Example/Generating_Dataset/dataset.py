import os
import sys
import numpy as np

from DeepPhysX_Core.Pipelines.BaseDataGenerator import BaseDataGenerator
from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX_Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig

from Environment import Environment

env_config = BaseEnvironmentConfig(environment_class=Environment,
                                   environment_file=sys.modules[Environment.__module__].__file__)
dataset_config = BaseDatasetConfig()
nb_batch, batch_size = 10, 10
pipeline = BaseDataGenerator(dataset_config=dataset_config,
                             environment_config=env_config,
                             session_name='example',
                             nb_batches=nb_batch,
                             batch_size=batch_size)
pipeline.execute()


# Check that the Dataset has been well saved
dataset = []
for filename in [f for f in os.listdir(os.getcwd() + '/example/dataset') if f.__contains__('.npy')]:
    filename = os.getcwd() + '/example/dataset/' + filename
    dataset.append([])
    with open(filename, 'rb') as file:
        for _ in range(nb_batch * batch_size):
            dataset[-1].append(np.load(file).tolist())
expected = [[[i, i] for i in range(nb_batch * batch_size)],
            [[2*i, 2*i] for i in range(nb_batch * batch_size)]]
if len(dataset) == 2 and dataset == expected:
    print("Dataset is healthy.")
