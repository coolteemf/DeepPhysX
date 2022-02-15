from unittest import TestCase
import os
import shutil
from numpy import array

from DeepPhysX_Core.Manager.DatasetManager import DatasetManager
from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig, BaseDataset


class TestDatasetManager(TestCase):

    def setUp(self):
        self.data_config = BaseDatasetConfig(shuffle_dataset=False)
        self.manager = None

    def tearDown(self):
        for folder in [f for f in os.listdir(os.getcwd()) if f.__contains__('dataset')]:
            if os.path.isdir(folder):
                shutil.rmtree(folder)

    def test_init(self):
        self.manager = DatasetManager(dataset_config=self.data_config,
                                      session_dir=os.getcwd())
        # Check default values
        self.assertIsInstance(self.manager.dataset, BaseDataset)
        self.assertEqual(self.manager.max_size, 1e9)
        self.assertEqual(self.manager.shuffle_dataset, False)
        self.assertFalse(False in self.manager.record_data.values())
        self.assertEqual(self.manager.mode, 0)
        self.assertEqual(len(self.manager.partitions_templates), 3)
        self.assertFalse(False in [len(field) == 1 for field in self.manager.fields.values()])
        self.assertFalse(False in [partitions == [[], [], []] for partitions in self.manager.list_partitions.values()])
        self.assertFalse(False in [idx == 0 for idx in self.manager.idx_partitions])
        self.assertFalse(False in [current is None for current in self.manager.current_partition_path.values()])
        for attribute in [self.manager.mul_part_idx, self.manager.mul_part_slices, self.manager.mul_part_list_path]:
            self.assertEqual(attribute, None)
        self.assertEqual(self.manager.dataset_dir, os.path.join(os.getcwd(), 'dataset/'))
        self.assertEqual(self.manager.new_session, True)
        # Check repository creation
        self.assertTrue(os.path.isdir(self.manager.dataset_dir))

    def test_add_data(self):
        self.manager = DatasetManager(dataset_config=self.data_config, session_dir=os.getcwd())
        # Add a batch
        data = {'input': array([[i] for i in range(10)]),
                'output': array([[2 * i] for i in range(10)])}
        self.manager.add_data(data)
        # Check dataset
        for field, value in data.items():
            self.assertTrue((self.manager.dataset.data[field] == value).all())
        # Check repository
        self.assertFalse(self.manager.first_add)
        self.assertFalse(False in [len(partitions[self.manager.mode]) == 1 for partitions in
                                   self.manager.list_partitions.values()])
        self.assertFalse(False in [current is not None for current in self.manager.current_partition_path.values()])
        self.assertEqual(self.manager.idx_partitions[self.manager.mode], 1)

    def test_get_data(self):
        self.manager = DatasetManager(dataset_config=self.data_config, session_dir=os.getcwd())
        # Add a batch
        data = {'input': array([[i] for i in range(10)]),
                'output': array([[2 * i] for i in range(10)])}
        self.manager.add_data(data)
        # Get a batch
        batch = self.manager.get_data(True, True, 5)
        for field, value in data.items():
            self.assertTrue((batch[field] == value[:5]).all())
        batch = self.manager.get_data(True, True, 5)
        for field, value in data.items():
            self.assertTrue((batch[field] == value[5:]).all())
        batch = self.manager.get_data(True, True, 8)
        for field, value in data.items():
            self.assertTrue((batch[field] == value[:8]).all())
        batch = self.manager.get_data(True, True, 8)
        for field, value in data.items():
            self.assertEqual(batch[field].tolist(), value[-2:].tolist() + value[:6].tolist())

    def test_register_new_fields(self):
        self.manager = DatasetManager(dataset_config=self.data_config, session_dir=os.getcwd())
        fields = {'IN': ['IN_field1'],
                  'OUT': ['OUT_field1', 'OUT_field2']}
        self.manager.register_new_fields(fields)
        # Check fields
        self.assertEqual(len(self.manager.fields['IN']), 1 + 1)
        self.assertEqual(len(self.manager.fields['OUT']), 1 + 2)
        self.assertFalse(False in [field in self.manager.list_partitions for field in fields['IN'] + fields['OUT']])
        self.assertFalse(False in [self.manager.list_partitions[field] == [[], [], []] for field in fields['IN'] +
                                   fields['OUT']])
        self.assertFalse(False in [self.manager.record_data[field] for field in fields['IN'] + fields['OUT']])
