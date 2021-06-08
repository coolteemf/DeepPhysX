import unittest
import numpy as np

from DeepPhysX.Dataset.BaseDataset import BaseDataset
from DeepPhysX.Dataset.BaseDatasetConfig import BaseDatasetConfig


class TestBaseDatasetConfig(unittest.TestCase):

    def setUp(self):
        pass

    def test_init(self):
        with self.assertRaises(TypeError):
            BaseDatasetConfig(dataset_dir=0)
            BaseDatasetConfig(dataset_dir=None, partition_size='1.0')
            BaseDatasetConfig(dataset_dir=None, partition_size=1., generate_data='True')
            BaseDatasetConfig(dataset_dir=None, partition_size=1., generate_data=True, shuffle_dataset='True')

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
            dataset.add([0., 0.], np.array([[0., 0.]]))
            dataset.add(np.array([[0., 0.]]), [0., 0.])
        with self.assertRaises(ValueError):
            dataset.add(np.array([0., 0.]), np.array([[0., 0.]]))
            dataset.add(np.array([[0., 0.]]), np.array([0., 0.]))
        data_in, data_out = [], []
        for i in range(6):
            data_in.append([i+1, i+1])
            data_out.append([i+1, i+1])
            dataset.add(np.array([data_in[-1]]), np.array([[data_out[-1]]]))
        self.assertTrue((dataset.data_in == np.array(data_in)).all())
        self.assertTrue((dataset.data_out == np.array(data_out)).all())
        self.assertEqual(dataset.memory_size(), (96, 96))

    def test_load(self):
        dataset = self.dataset_config.createDataset()
        with self.assertRaises(TypeError):
            dataset.load([0., 0.], np.array([[0., 0.]]))
            dataset.load(np.array([[0., 0.]]), [0., 0.])
        with self.assertRaises(ValueError):
            dataset.load(np.array([0., 0.]), np.array([[0., 0.]]))
            dataset.load(np.array([[0., 0.]]), np.array([0., 0.]))
        data_in, data_out = [], []
        for i in range(6):
            data_in.append([i+1, i+1])
            data_out.append([i+1, i+1])
        dataset.load(np.array(data_in), np.array(data_out))
        self.assertTrue((dataset.data_in == np.array(data_in)).all())
        self.assertTrue((dataset.data_out == np.array(data_out)).all())
        self.assertEqual(dataset.memory_size(), (96, 96))

    def test_reset(self):
        dataset = self.dataset_config.createDataset()
        dataset.add(np.array([[i, i+1] for i in range(10)]), np.array([[i, i+1] for i in range(10)]))
        self.assertEqual(dataset.memory_size(), (160, 160))
        dataset.reset()
        self.assertEqual(dataset.memory_size(), (0, 0))

    def test_shuffle(self):
        dataset = self.dataset_config.createDataset()
        data = []
        for i in range(10):
            for j in range(10):
                data.append([i+1, j+1])
                dataset.add(np.array([data[-1]]), np.array([data[-1]]))
        dataset.shuffle()
        self.assertFalse((dataset.data_in == np.array(data)).all())


class TestDatasetManager(unittest.TestCase):

    def setUp(self):
        self.dataset_config = BaseDatasetConfig()

    # Todo: Do network objects first then go to managers which manage storage
