from os.path import join as osPathJoin
from os.path import isfile, isdir
from os import listdir

from json import dump as json_dump
from json import load as json_load
from typing import Any, Dict, Tuple, List, Optional, Union

import numpy
import numpy as np
from numpy import load, squeeze

from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX_Core.Utils.pathUtils import get_first_caller, create_dir
from DeepPhysX_Core.Utils.jsonUtils import CustomJSONEncoder


class DatasetManager:

    def __init__(self,
                 dataset_config: BaseDatasetConfig,
                 data_manager: Any = None,
                 session_name: str = 'default',
                 session_dir: str = None,
                 new_session: bool = True,
                 train: bool = True,
                 record_data: Optional[Dict[str, bool]] = None):

        """
        DatasetManager handle all operations with input / output files. Allows saving and read tensors from files.

        :param BaseDatasetConfig dataset_config: Specialisation containing the parameters of the dataset manager
        :param DataManager data_manager: DataManager that handles the DatasetManager
        :param str session_name: Name of the newly created directory if session_dir is not defined
        :param str session_dir: Name of the directory in which to write all the necessary data
        :param bool new_session: Define the creation of new directories to store data
        :param bool train: True if this session is a network training
        :param dict record_data: Format {\'in\': bool, \'out\': bool} save the tensor when bool is True
        """

        self.name: str = self.__class__.__name__
        self.data_manager: Any = data_manager

        # Checking arguments
        if dataset_config is not None and not isinstance(dataset_config, BaseDatasetConfig):
            raise TypeError(f"[{self.name}] The dataset config must be a BaseDatasetConfig object.")
        if type(session_name) != str:
            raise TypeError(f"[{self.name}] The session name must be a str.")
        if session_dir is not None and type(session_dir) != str:
            raise TypeError(f"[{self.name}] The session directory must be a str.")
        if type(new_session) != bool:
            raise TypeError(f"[{self.name}] The 'new_network' argument must be a boolean.")
        if type(train) != bool:
            raise TypeError(f"[{self.name}] The 'train' argument must be a boolean.")
        if record_data is not None and type(record_data) != dict:
            raise TypeError(f"[{self.name}] The 'record_data' argument must be a dict.")
        elif record_data is not None:
            if type(record_data['input']) != bool or type(record_data['output']) != bool:
                raise TypeError(f"[{self.name}] The values of 'record_data' must be booleans.")

        # Create the Dataset object (default if no config specified)
        dataset_config = BaseDatasetConfig() if dataset_config is None else dataset_config
        self.dataset = dataset_config.create_dataset()

        # Dataset parameters
        self.max_size: int = self.dataset.max_size
        self.shuffle_dataset: bool = dataset_config.shuffle_dataset
        self.record_data: Dict[str, bool] = record_data if record_data is not None else {'input': True, 'output': True}
        self.first_add: bool = True

        # Dataset modes
        self.modes: Dict[str, int] = {'Training': 0, 'Validation': 1, 'Running': 2}
        self.mode: int = self.modes['Training'] if train else self.modes['Running']
        self.last_loaded_dataset_mode: int = self.mode

        # Dataset partitions
        self.partitions_templates: Tuple[str, str, str] = (session_name + '_training_{}_{}.npy',
                                                           session_name + '_validation_{}_{}.npy',
                                                           session_name + '_running_{}_{}.npy')
        self.fields: Dict[str, List[str]] = {'IN': ['input'], 'OUT': ['output']}
        self.list_partitions: Dict[str, Optional[List[List[numpy.ndarray]]]] = {
            'input': [[], [], []] if self.record_data['input'] else None,
            'output': [[], [], []] if self.record_data['output'] else None}
        self.idx_partitions: List[int] = [0, 0, 0]
        self.current_partition_path: Dict[str, Optional[str]] = {'input': None, 'output': None}

        # Dataset loading with multiple partitions variables
        self.mul_part_list_path: Optional[Dict[int, Dict[str, List[str]]]] = None
        self.mul_part_slices: Optional[List[List[int]]] = None
        self.mul_part_idx: Optional[int] = None

        # Dataset Json file
        self.json_filename: str = 'dataset.json'
        self.json_empty: Dict[str, Dict[str, Union[List[int], Dict[Any, Any]]]] = {'data_shape': {},
                                                                                   'nb_samples': {mode: [] for mode in
                                                                                                  self.modes},
                                                                                   'partitions': {mode: {} for mode in
                                                                                                  self.modes}}
        self.json_dict: Dict[str, Dict[str, Union[List[int], Dict[Any, Any]]]] = self.json_empty.copy()
        self.json_found: bool = False

        # Dataset repository
        self.session_dir: str = session_dir if session_dir is not None else osPathJoin(get_first_caller(), session_name)
        dataset_dir: str = dataset_config.dataset_dir
        self.new_session: bool = new_session
        self.__new_dataset: bool = False

        # Training
        if train:
            # New training session
            if new_session:
                # New training session with new dataset
                if dataset_dir is None:
                    self.dataset_dir: str = create_dir(dir_path=osPathJoin(self.session_dir, 'dataset/'),
                                                  dir_name='dataset')
                    self.__new_dataset = True
                # New training session with existing dataset
                else:
                    if dataset_dir[-1] != "/":
                        dataset_dir += "/"
                    if dataset_dir[-8:] != "dataset/":
                        dataset_dir += "dataset/"
                    self.dataset_dir = dataset_dir
                    self.load_directory()
            # Existing training session
            else:
                self.dataset_dir = osPathJoin(self.session_dir, 'dataset/')
                self.load_directory()
        # Prediction
        else:
            self.dataset_dir = osPathJoin(self.session_dir, 'dataset/')
            self.__new_dataset = True
            self.create_running_partitions()

    def get_data_manager(self) -> Any:
        """
        Return the Manager of the DataManager.

        :return: DataManager that handle The DatasetManager
        """
        return self.data_manager

    def add_data(self, data: Dict[str, Union[numpy.ndarray, Dict[str, numpy.ndarray]]]) -> None:
        """
        Push the data in the dataset. If max size is reached generate a new partition and write into it.

        :param dict data: Format {'input':numpy.ndarray, 'output':numpy.ndarray}  contain in 'input' input tensors and
        in 'output' output tensors.

        :return:
        """

        # 1. If first add, create first partitions
        if self.first_add:
            new_fields = {}
            for side, key in zip(['IN', 'OUT'], ['dataset_in', 'dataset_out']):
                if key in data:
                    new_fields[side] = [f'{side}_{field}' for field in data[key].keys()]
            self.register_new_fields(new_fields)
            self.create_partitions()

        # 2. Add network data
        for field in ['input', 'output']:
            if self.record_data[field]:
                self.dataset.add(field, data[field], self.current_partition_path[field])

        # 3. Add additional data
        # 3.1 If there is additional data, convert field names then add each field
        if 'dataset_in' in data.keys() or 'dataset_out' in data.keys():
            for side, key in zip(['IN', 'OUT'], ['dataset_in', 'dataset_out']):
                additional_data = {}
                if key in data:
                    additional_data = {f'{side}_{field}': data[key][field] for field in data[key].keys()}
                # Check all registered fields are in additional data
                for field in self.fields[side][1:]:
                    if field not in additional_data:
                        raise ValueError(f"[{self.name}] No data received for the additional field {field}.")
                # Add each field to dataset
                for field in additional_data:
                    self.dataset.add(field, additional_data[field], self.current_partition_path[field])
        # 3.2 If there is no additional data but registered additional data
        elif 'dataset_in' not in data.keys() and len(self.fields['IN']) > 1:
            raise ValueError(f"[{self.name}] No data received for the additional fields {self.fields['IN']}")
        elif 'dataset_out' not in data.keys() and len(self.fields['OUT']) > 1:
            raise ValueError(f"[{self.name}] No data received for the additional fields {self.fields['OUT']}")

        # 4. Update json file
        self.update_json(update_nb_samples=True)
        if self.first_add:
            self.update_json(update_partitions_lists=True, update_shapes=True)
            self.first_add = False

        # 5. Check the size of the dataset
        if self.dataset.memory_size() > self.max_size:
            self.save_data()
            self.create_partitions()
            self.update_json(update_partitions_lists=True, update_nb_samples=True)
            self.dataset.empty()

    def get_data(self, get_inputs: bool, get_outputs: bool, batch_size: int = 1, batched: bool = True) -> Dict[str, numpy.ndarray]:
        """
        Fetch tensors from the dataset or reload partitions if dataset is empty or specified.

        :param bool get_inputs: If True fill the data['input'] field
        :param bool get_outputs: If True fill the data['output'] field
        :param int batch_size: Size of a batch
        :param bool batched: Add an empty dimension before [4,100] -> [0,4,100]

        :return: dict of format {'input':numpy.ndarray, 'output':numpy.ndarray} filled with desired data
        """

        # 1. Check if a dataset is loaded and if the current sample is not the last
        if self.current_partition_path['input'] is None or self.dataset.current_sample >= self.dataset.nb_samples:
            # if not force_partition_reload:
            #     return None
            self.load_partitions()
            if self.shuffle_dataset:
                self.dataset.shuffle()
            self.dataset.current_sample = 0

        # 2. Update dataset indices with batch size
        idx = self.dataset.current_sample
        self.dataset.current_sample += batch_size

        # 3. Get a batch of each data field if input / output sides are required
        data = {}
        fields = self.fields['IN'][:] if get_inputs else []
        fields += self.fields['OUT'][:] if get_outputs else []
        for field in fields:
            # Network input and output fields
            if field in ['input', 'output']:
                data[field] = self.dataset.get(field, idx, idx + batch_size)
                if not batched:
                    data[field] = squeeze(data[field], axis=0)
            # Additional data fields
            else:
                side = 'dataset_in' if field[:3] == 'IN_' else 'dataset_out'
                if side not in data.keys():
                    data[side] = {}
                user_field = field[3:] if side == 'dataset_in' else field[4:]
                data[side][user_field] = self.dataset.get(field, idx, idx + batch_size)
                if not batched:
                    data[side][user_field] = squeeze(data[side][user_field], axis=0)

        # 4. Ensure each field received the same batch size
        if data['input'].shape[0] != data['output'].shape[0]:
            raise ValueError(f"[{self.name}] Size of loaded batch mismatch for input and output "
                             f"(in: {data['input'].shape} / out: {data['output'].shape}")
        for data_side in ['dataset_in', 'dataset_out']:
            if data_side in data.keys():
                for field in data[data_side].keys():
                    if data[data_side][field].shape[0] != data['input'].shape[0]:
                        raise ValueError(f"[{self.name}] Size of loaded batch mismatch for {data_side} field {field} "
                                         f"(net: {data['input'].shape} / {field}: {data[data_side][field].shape}")

        # 5. Ensure the batch has the good size, otherwise load new data to complete it
        if data['input'].shape[0] < batch_size:
            # Load next samples from the dataset
            self.load_partitions()
            if self.shuffle_dataset:
                self.dataset.shuffle()
            self.dataset.current_sample = 0
            # Get the remaining samples
            missing_samples = batch_size - data['input'].shape[0]
            missing_data = self.get_data(get_inputs=get_inputs, get_outputs=get_outputs, batch_size=missing_samples)
            # Merge fields
            for field in ['input', 'output']:
                data[field] = np.concatenate((data[field], missing_data[field]))
            for side in ['dataset_in', 'dataset_out']:
                if side in data.keys():
                    for field in data[side].keys():
                        data[side][field] = np.concatenate((data[side][field], missing_data[side][field]))
        return data

    def register_new_fields(self, new_fields: Dict[str, str]) -> None:
        """
        Add new data fields in the dataset.

        :param dict new_fields: Name of the new fields split in either 'IN' side or 'OUT' side of the dataset
        :return:
        """

        for side in ['IN', 'OUT']:
            if side in new_fields:
                for field in new_fields[side]:
                    self.register_new_field(side, field)

    def register_new_field(self, side: str, new_field: str) -> None:
        """
        Add a new data field in the dataset.

        :param side: Either 'IN' or 'OUT' side of the Dataset.
        :param new_field: Name of the new field.
        :return:
        """

        if new_field not in self.fields[side]:
            self.fields[side].append(new_field)
            self.list_partitions[new_field] = [[], [], []]
            self.record_data[new_field] = True

    def create_partitions(self) -> None:
        """
        Create a new partition for current mode and for each registered fields.

        :return:
        """

        print(f"[{self.name}] New partitions added for each field with max size ~{float(self.max_size) / 1e9}Gb.")
        for side in self.fields:
            for field in self.fields[side]:
                if self.record_data[field]:
                    name = side if field in ['input', 'output'] else field
                    partition_path = self.partitions_templates[self.mode].format(name,
                                                                                 self.idx_partitions[self.mode])
                    self.list_partitions[field][self.mode].append(partition_path)
                    self.current_partition_path[field] = self.dataset_dir + partition_path
        self.idx_partitions[self.mode] += 1

    def create_running_partitions(self) -> None:
        """
        Run specific function. Handle partitions creation when not training.

        :return:
        """

        # 1. Load the directory without loading data
        self.load_directory(load_data=False)

        # 2. Find out how many partitions exists for running mode
        mode = list(self.modes.keys())[self.mode]
        partitions_dict = self.json_dict['partitions'][mode]
        nb_running_partitions = max([len(partitions_dict[field]) for field in partitions_dict.keys()])

        # 3. Create a new partition partitions
        self.idx_partitions[self.mode] = nb_running_partitions
        self.create_partitions()

    def load_directory(self, load_data: bool = True) -> None:
        """
        Load the desired directory. Try to find partition list and upload it.
        No data loading here.

        :return:
        """

        # 1. Check the directory exists
        if not isdir(self.dataset_dir):
            raise Warning(f"[{self.name}] Loading directory: The given path is not an existing directory")
        if load_data:
            print(f"[{self.name}] Loading directory: Read dataset from {self.dataset_dir}")

        # 2. Look for the json info file
        if isfile(osPathJoin(self.dataset_dir, self.json_filename)):
            self.json_found = True
            with open(osPathJoin(self.dataset_dir, self.json_filename)) as json_file:
                self.json_dict = json_load(json_file)

        # 3. Load partitions for each mode
        for mode in self.modes:
            # 3.1. Get sorted partitions
            partitions_dict = self.json_dict['partitions'][mode] if self.json_found else self.search_partitions(mode)
            # 3.2. Register additional fields
            for side in ['IN', 'OUT']:
                for field in partitions_dict:
                    if field.__contains__(side):
                        self.register_new_field(side, field)
            # 3.3. Register each partition
            for field in partitions_dict:
                self.list_partitions[field][self.modes[mode]] = partitions_dict[field]
            # 3.4. Check that the number of partitions is the same for each field
            number_of_partitions = len(self.list_partitions[self.fields['IN'][0]][self.modes[mode]])
            for field in self.fields['IN'] + self.fields['OUT']:
                if len(self.list_partitions[field][self.modes[mode]]) != number_of_partitions:
                    raise ValueError(f"[{self.name}] The number of partitions is different for {field} with "
                                     f"{len(self.list_partitions[field][self.modes[mode]])} partitions found.")

        # 4. Update Json file if not found or partially empty
        if not self.json_found or self.empty_json_fields():
            self.search_partitions_info()
            self.update_json(update_partitions_lists=True)

        # 4. Load data from partitions
        if load_data:
            self.idx_partitions = [len(partitions_list) for partitions_list in self.list_partitions['input']]
            self.load_partitions(force_reload=True)

    def search_partitions(self, mode: str) -> Dict[str, List[str]]:
        """
        If loading a directory without JSON info file, search for existing partitions manually.

        :param str mode: Mode of the partitions to find.
        :return:
        """

        # 1. Get all the partitions for the mode
        partitions_dict = {}
        partitions_list = [f for f in listdir(self.dataset_dir) if isfile(osPathJoin(self.dataset_dir, f))
                           and f.endswith('.npy') and f.__contains__(mode.lower())]

        # 2. Sort partitions by side (IN, OUT) and by name
        in_partitions = sorted([file for file in partitions_list if file.__contains__('_IN_')])
        out_partitions = sorted([file for file in partitions_list if file.__contains__('_OUT_')])

        # 3. Sort partitions by data field
        for side, partitions, net_field in zip(['IN', 'OUT'], [in_partitions, out_partitions], ['input', 'output']):
            for partition in partitions:
                # Extract information from the filenames
                split_name = partition.split('_')
                clues = split_name[split_name.index(side):]
                # Partition name for network data: {NAME_OF_SESSION}_SIDE_IDX.npy
                if len(clues) == 2:
                    if net_field not in partitions_dict.keys():
                        partitions_dict[net_field] = []
                    partitions_dict[net_field].append(partition)
                # Partition name for additional data: {NAME_OF_SESSION}_SIDE_{NAME_OF_FIELD}_IDX.npy
                else:
                    field_name = '_'.join(clues[:-1])
                    if field_name not in partitions_dict.keys():
                        partitions_dict[field_name] = []
                    partitions_dict[field_name].append(partition)

        return partitions_dict

    def search_partitions_info(self) -> None:
        """
        If loading a directory without JSON info file

        :return:
        """

        # 1. Get the shape of each partition
        partition_shapes = [{field: [] for field in self.fields['IN'] + self.fields['OUT']} for _ in self.modes]
        for mode in self.modes:
            for field in self.fields['IN'] + self.fields['OUT']:
                for partition in [self.dataset_dir + path for path in self.list_partitions[field][self.modes[mode]]]:
                    partition_data = np.load(partition)
                    partition_shapes[self.modes[mode]][field].append(partition_data.shape)
                    del partition_data

        # 2. Get the number of samples per partition for each mode
        for mode in self.modes:
            number_of_samples = {}
            for field in self.fields['IN'] + self.fields['OUT']:
                number_of_samples[field] = [shape[0] for shape in partition_shapes[self.modes[mode]][field]]
                # Number of samples through partitions must be the same along fields
                if number_of_samples[field] != list(number_of_samples.values())[0]:
                    raise ValueError(f"[{self.name}] The number of sample in each partition is not consistent:\n"
                                     f"{number_of_samples}")
            # Store the number of samples per partition for the mode
            self.json_dict['nb_samples'][mode] = list(number_of_samples.values())[0]

        # 3. Get the data shape for each field
        data_shape = {field: [] for field in self.fields['IN'] + self.fields['OUT']}
        for mode in self.modes:
            for field in self.fields['IN'] + self.fields['OUT']:
                for i, shape in enumerate(partition_shapes[self.modes[mode]][field]):
                    if len(data_shape[field]) == 0:
                        data_shape[field] = shape[1:]
                    # Data shape must be the same along partitions and mode
                    if shape[1:] != data_shape[field]:
                        raise ValueError(f"[{self.name}] Two different data sizes found for mode {mode}, field {field},"
                                         f" partition n°{i}: {data_shape[field]} vs {shape[1:]}")
        # Store the data shapes
        self.json_dict['data_shape'] = data_shape

    def load_partitions(self, force_reload: bool = False) -> None:
        """
        Load data from partitions.

        :param force_reload:
        :return:
        """

        # 1. If there is only one partition for the current mode for input field at least, don't need to reload it
        if self.last_loaded_dataset_mode == self.mode and self.idx_partitions[self.mode] == 1 and not force_reload:
            print("LOAD PARTITION SKIP")
            return

        # 2. Check partitions existence for the current mode
        if self.idx_partitions[self.mode] == 0:
            raise ValueError(f"[{self.name}] No partitions to read for {list(self.modes.keys())[self.mode]} mode.")

        # 3. Load new data in dataset
        self.dataset.empty()
        # Training mode with mixed dataset: read multiple partitions per field
        if self.mode == self.modes['Training'] and self.idx_partitions[self.modes['Running']] > 0:
            if self.mul_part_idx is None:
                self.load_multiple_partitions([self.modes['Training'], self.modes['Running']])
            self.read_multiple_partitions()
            return
        # Training mode without mixed dataset or other modes: check the number of partitions per field to read
        if self.idx_partitions[self.mode] == 1:
            self.read_last_partitions()
        else:
            if self.mul_part_idx is None:
                self.load_multiple_partitions([self.mode])
            self.read_multiple_partitions()

    def read_last_partitions(self) -> None:
        """
        Load the last partitions for each data field.

        :return:
        """

        for field in self.fields['IN'] + self.fields['OUT']:
            self.current_partition_path[field] = self.dataset_dir + self.list_partitions[field][self.mode][-1]
            data = load(self.current_partition_path[field])
            self.dataset.set(field, data)

    def load_multiple_partitions(self, modes: List[int]) -> None:
        """
        Specialisation of the load_partitions() function. It can load a list of partitions

        :param list modes: Recommended to use datasetManager.modes['name_of_desired_mode'] in order to correctly load
        the dataset
        :return:
        """

        # 1. Initialize multiple partition loading variables
        self.mul_part_list_path = {field: [] for field in self.fields['IN'] + self.fields['OUT']}
        self.mul_part_slices = []
        self.mul_part_idx = 0
        nb_sample_per_partition = {field: [] for field in self.fields['IN'] + self.fields['OUT']}

        # 2. For each field, load all partitions
        for field in self.fields['IN'] + self.fields['OUT']:
            for mode in modes:
                # 2.1. Add partitions to the list of partitions to read
                self.mul_part_list_path[field] += [self.dataset_dir + partition
                                                   for partition in self.list_partitions[field][mode]]
                # 2.2. Find the number of samples in each partition
                nb_sample_per_partition[field] += self.json_dict['nb_samples'][list(self.modes.keys())[mode]]

        # 3. Invert the partitions list structure
        nb_partition = len(nb_sample_per_partition[self.fields['IN'][0]])
        inverted_list = [{} for _ in range(nb_partition)]
        for i in range(nb_partition):
            for field in self.fields['IN'] + self.fields['OUT']:
                inverted_list[i][field] = self.mul_part_list_path[field][i]
        self.mul_part_list_path = inverted_list

        # 4. Define the slicing pattern of reading for partitions
        for idx in nb_sample_per_partition[self.fields['IN'][0]]:
            idx_slicing = [0]
            for _ in range(nb_partition - 1):
                idx_slicing.append(idx_slicing[-1] + idx // nb_partition + 1)
            idx_slicing.append(idx)
            self.mul_part_slices.append(idx_slicing)

    def read_multiple_partitions(self) -> None:
        """
        Read data in a list of partitions.

        :return:
        """

        for i, partitions in enumerate(self.mul_part_list_path):
            for field in partitions.keys():
                dataset = np.load(partitions[field])
                samples = slice(self.mul_part_slices[i][self.mul_part_idx],
                                self.mul_part_slices[i][self.mul_part_idx + 1])
                self.dataset.add(field, dataset[samples])
                del dataset
        self.mul_part_idx = (self.mul_part_idx + 1) % (len(self.mul_part_slices[0]) - 1)
        self.current_partition_path['input'] = self.mul_part_list_path[0][self.fields['IN'][0]]
        self.dataset.current_sample = 0

    def update_json(self, update_shapes: bool = False, update_nb_samples: bool = False, update_partitions_lists: bool = False) -> None:
        """
        Update the json info file with the current Dataset repository information.

        :param bool update_shapes: If True, data shapes per field are overwritten
        :param bool update_nb_samples: If True, number of samples per partition are overwritten
        :param bool update_partitions_lists: If True, list of partitions is overwritten
        :return:
        """

        # Update data shapes
        if update_shapes:
            for field in self.fields['IN'] + self.fields['OUT']:
                self.json_dict['data_shape'][field] = self.dataset.get_data_shape(field)

        # Update number of samples
        if update_nb_samples:
            idx_mode = list(self.modes.keys())[self.mode]
            if len(self.json_dict['nb_samples'][idx_mode]) == self.idx_partitions[self.mode]:
                self.json_dict['nb_samples'][idx_mode][-1] = self.dataset.current_sample
            else:
                self.json_dict['nb_samples'][idx_mode].append(self.dataset.current_sample)

        # Update partitions lists
        if update_partitions_lists:
            for mode in self.modes:
                for field in self.fields['IN'] + self.fields['OUT']:
                    self.json_dict['partitions'][mode][field] = self.list_partitions[field][self.modes[mode]]

        # Overwrite json file
        with open(self.dataset_dir + self.json_filename, 'w') as json_file:
            json_dump(self.json_dict, json_file, indent=3, cls=CustomJSONEncoder)

    def empty_json_fields(self) -> bool:
        """
        Check if the json info file contains empty fields.

        :return:
        """
        for key in self.json_empty:
            if self.json_dict[key] == self.json_empty[key]:
                return True
        return False

    def save_data(self) -> None:
        """
        Close all open files

        :return:
        """
        if self.__new_dataset:
            for field in self.current_partition_path.keys():
                self.dataset.save(field, self.current_partition_path[field])

    def set_mode(self, mode) -> None:
        """
        Set the DatasetManager working mode.

        :param int mode: Recommended to use datasetManager_modes['name_of_desired_mode'] in order to correctly set up
        the DatasetManager

        :return:
        """
        # Nothing has to be done if you do not change mode
        if mode == self.mode:
            return
        if self.mode == self.modes['Running']:
            print(f"[{self.name}] It's not possible to switch dataset mode while running.")
        else:
            # Save dataset before changing mode
            self.save_data()
            self.mode = mode
            self.dataset.empty()
            # Create or load partition for the new mode
            if self.idx_partitions[self.mode] == 0:
                print(f"[{self.name}] Change to {self.mode} mode, create a new partition")
                self.create_partitions()
            else:
                print(f"[{self.name}] Change to {self.mode} mode, load last partition")
                self.read_last_partitions()

    def new_dataset(self) -> bool:
        return self.__new_dataset

    def getNextBatch(self, batch_size: int) -> Dict[str, numpy.ndarray]:
        """
        :param int batch_size: Size of the batch
        :return: dict of format {'input': numpy.ndarray, 'output': numpy.ndarray} filled with a batch of data
        """
        return self.get_data(get_inputs=True, get_outputs=True, batch_size=batch_size, batched=True)

    def getNextSample(self, batched: bool = True) -> Dict[str, numpy.ndarray]:
        """
        :return: dict of format {'input': numpy.ndarray, 'output': numpy.ndarray} filled with a sample of data
        """
        return self.get_data(get_inputs=True, get_outputs=True, batched=batched)

    def getNextInput(self, batched: bool = False) -> Dict[str, numpy.ndarray]:
        """
        :return: dict of format {'input': numpy.ndarray, 'output': numpy.ndarray} where only the input field is filled
        """
        return self.get_data(get_inputs=True, get_outputs=False, batched=batched)

    def getNextOutput(self, batched: bool = False) -> Dict[str, numpy.ndarray]:
        """
        :return: dict of format {'input': numpy.ndarray, 'output': numpy.ndarray} where only the output field is filled
        """
        return self.get_data(get_inputs=False, get_outputs=True, batched=batched)

    def close(self) -> None:
        """
        Launch the close procedure of the dataset manager

        :return:
        """
        self.save_data()

    def __str__(self) -> str:
        """
        :return: A string containing valuable information about the DatasetManager
        """
        description = "\n"
        description += f"# {self.name}\n"
        description += f"    Dataset Repository: {self.dataset_dir}\n"
        description += f"    Partitions size: {self.max_size * 1e-9} Go\n"
        description += f"    Managed objects: Dataset: {self.dataset.name}\n"
        description += str(self.dataset)
        return description
