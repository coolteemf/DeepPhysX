import os
import unittest
import numpy as np
from tempfile import TemporaryFile

from DeepPhysX_Core.Dataset.BaseDataset import BaseDataset
from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig


class TestBaseDatasetConfig(unittest.TestCase):

    def setUp(self):
        pass

    def test_init(self):
        # TypeError
        with self.assertRaises(TypeError):
            BaseDatasetConfig(dataset_dir=0)
            BaseDatasetConfig(partition_size='1.0')
            BaseDatasetConfig(shuffle_dataset='True')
        # ValueError
        with self.assertRaises(ValueError):
            BaseDatasetConfig(dataset_dir=os.path.join(os.getcwd(), 'dataset'))
            BaseDatasetConfig(partition_size=0)

    def test_createDataset(self):
        # ValueError
        class Test1:
            def __init__(self):
                pass
        dataset_config = BaseDatasetConfig(dataset_class=Test1)
        self.assertRaises(ValueError, dataset_config.createDataset)
        # TypeError
        class Test2:
            def __init__(self, config):
                pass
        dataset_config = BaseDatasetConfig(dataset_class=Test2)
        self.assertRaises(TypeError, dataset_config.createDataset)
        # No error
        class Test3(BaseDataset):
            def __init__(self, config):
                BaseDataset.__init__(self, config)
        dataset_config = BaseDatasetConfig(dataset_class=Test3)
        self.assertIsInstance(dataset_config.createDataset(), BaseDataset)


class TestBaseDataset(unittest.TestCase):

    def setUp(self):
        dataset_config = BaseDatasetConfig()
        self.dataset = dataset_config.createDataset()

    def test_add(self):
        # TypeError
        with self.assertRaises(TypeError):
            self.dataset.add('in', [0., 0.], None)
        # Adding data by sample
        data_in, data_out = [], []
        file_in, file_out = TemporaryFile(), TemporaryFile()
        for _ in range(3):
            data_in.append([0, 0])
            data_out.append([0, 0])
            self.dataset.add('in', np.array([data_in[-1]]), file_in)
            self.dataset.add('out', np.array([data_out[-1]]), file_out)
        self.assertTrue((self.dataset.data_in == np.array(data_in)).all())
        self.assertTrue((self.dataset.data_out == np.array(data_out)).all())
        # Adding data by batch
        for _ in range(3):
            data_in.append([0, 0])
            data_out.append([0, 0])
        self.dataset.add('in', np.array(data_in[3:]), file_in)
        self.dataset.add('out', np.array(data_out[3:]), file_out)
        self.assertTrue((self.dataset.data_in == np.array(data_in)).all())
        self.assertTrue((self.dataset.data_out == np.array(data_out)).all())
        # Check size
        self.assertEqual(self.dataset.current_sample, 6)
        self.assertEqual(self.dataset.memory_size(), 192)
        # Close temporary files
        file_in.close()
        file_out.close()

    def test_load(self):
        # TypeError
        with self.assertRaises(TypeError):
            self.dataset.load('in', [0., 0.])
        # Adding data by sample
        data_in, data_out = [], []
        for _ in range(6):
            data_in.append([0, 0])
            data_out.append([0, 0])
            self.dataset.load('in', np.array(data_in[-1]))
            self.dataset.load('out', np.array(data_out[-1]))
        self.assertTrue((self.dataset.data_in == np.array(data_in)).all())
        self.assertTrue((self.dataset.data_out == np.array(data_out)).all())
        # Check size
        self.assertEqual(self.dataset.current_sample, 6)
        self.assertEqual(self.dataset.memory_size(), 192)

    def test_reset(self):
        # Adding data
        for _ in range(10):
            self.dataset.load('in', np.array([0, 0]))
            self.dataset.load('out', np.array([0, 0]))
        # Check size before and after reset
        self.assertEqual(self.dataset.current_sample, 10)
        self.assertEqual(self.dataset.memory_size(), 320)
        self.dataset.reset()
        self.assertEqual(self.dataset.current_sample, 0)
        self.assertEqual(self.dataset.memory_size(), 0)

    def test_shuffle(self):
        # Adding enough data to minimize the probability to get an identity shuffle
        data = []
        for i in range(10):
            for j in range(10):
                data.append([i, j])
                self.dataset.load('in', np.array(data[-1]))
        # Shuffle
        self.dataset.shuffle()
        self.assertFalse((self.dataset.data_in == np.array(data)).all())
