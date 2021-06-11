import unittest
import numpy as np
from tempfile import TemporaryFile

from DeepPhysX.Dataset.BaseDataset import BaseDataset
from DeepPhysX.Dataset.BaseDatasetConfig import BaseDatasetConfig


class TestBaseDatasetConfig(unittest.TestCase):

    def setUp(self):
        pass

    def test_init(self):
        with self.assertRaises(TypeError):
            BaseDatasetConfig(dataset_dir=0)
            BaseDatasetConfig(partition_size='1.0')
            BaseDatasetConfig(shuffle_dataset='True')

    def test_createDataset(self):
        # Bad
        config = BaseDatasetConfig(dataset_class=BaseDatasetConfig)
        self.assertRaises(TypeError, config.createDataset)
        # Good
        config = BaseDatasetConfig(dataset_class=BaseDataset)
        self.assertIsInstance(config.createDataset(), BaseDataset)


class TestBaseDataset(unittest.TestCase):

    def setUp(self):
        self.dataset_config = BaseDatasetConfig()

    def test_add(self):
        dataset = self.dataset_config.createDataset()
        with self.assertRaises(TypeError):
            dataset.add(None, [0., 0.], None)
        data_in, data_out = [], []
        file_in, file_out = TemporaryFile(), TemporaryFile()
        for i in range(6):
            data_in.append([i+1, i+1])
            data_out.append([i+1, i+1])
            dataset.add('in', np.array([data_in[-1]]), file_in)
            dataset.add('out', np.array([[data_out[-1]]]), file_out)
        self.assertTrue((dataset.data_in == np.array(data_in)).all())
        self.assertTrue((dataset.data_out == np.array(data_out)).all())
        self.assertEqual(dataset.memory_size(), 192)
        file_in.close()
        file_out.close()

    def test_load(self):
        dataset = self.dataset_config.createDataset()
        with self.assertRaises(TypeError):
            dataset.load(None, [0., 0.])
        data_in, data_out = [], []
        for i in range(6):
            data_in.append([i+1, i+1])
            data_out.append([i+1, i+1])
            dataset.load('in', np.array(data_in[-1]))
            dataset.load('out', np.array(data_out[-1]))
        self.assertTrue((dataset.data_in == np.array(data_in)).all())
        self.assertTrue((dataset.data_out == np.array(data_out)).all())
        self.assertEqual(dataset.memory_size(), 192)

    def test_reset(self):
        dataset = self.dataset_config.createDataset()
        file_in, file_out = TemporaryFile(), TemporaryFile()
        dataset.add('in', np.array([[i, i + 1] for i in range(10)]), file_in)
        dataset.add('out', np.array([[i, i + 1] for i in range(10)]), file_out)
        file_in.close()
        file_out.close()
        self.assertEqual(dataset.memory_size(), 320)
        dataset.reset()
        self.assertEqual(dataset.memory_size(), 0)

    def test_shuffle(self):
        dataset = self.dataset_config.createDataset()
        file_in, file_out = TemporaryFile(), TemporaryFile()
        data = []
        for i in range(10):
            for j in range(10):
                data.append([i+1, j+1])
                dataset.add('in', np.array([data[-1]]), file_in)
                dataset.add('out', np.array([data[-1]]), file_out)
        dataset.shuffle()
        file_in.close()
        file_out.close()
        self.assertFalse((dataset.data_in == np.array(data)).all())


class TestDatasetManager(unittest.TestCase):

    def setUp(self):
        self.dataset_config = BaseDatasetConfig()

    # Todo: Do network objects first then go to managers which manage storage
