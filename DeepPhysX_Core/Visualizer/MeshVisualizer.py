from vedo import buildPalette, Plotter, Mesh
from os.path import join as osPathJoin
from os import makedirs
from numpy import array, concatenate

from sys import maxsize as MAX_INT

from DeepPhysX_Core.Visualizer.VedoVisualizer import VedoVisualizer


class MeshVisualizer(VedoVisualizer):

    def __init__(self,
                 title='VedoVisualizer',
                 interactive_window=False,
                 show_axes=False,
                 min_color='yellow',
                 max_color='red',
                 range_color=10):
        """
        Display added objects. Mostly used to visualize data during training or prediction phase directly from the network.

        :param str title: Name of the window
        :param bool interactive_window: If True visualizer will be interactive (mouse actions)
        :param bool show_axes: If True display axes
        :param str min_color: Color of the min values of a mesh. Can be described as a list of 3 or 4 float using
        RGB and RGBA convention.
        :param str max_color: Color of the max values of a mesh. Can be described as a list of 3 or 4 float using
        RGB and RGBA convention.
        :param int range_color: Number of interpolations between min and max color
        """
        self.data = {}
        self.models = []
        self.models_shapes = {}
        self.viewer = None
        self.colormap = buildPalette(color1=min_color, color2=max_color, N=range_color, hsv=False)
        self.nb_view = 0
        self.params = {'title': title, 'interactive': interactive_window, 'axes': show_axes}
        # Wrong samples parameters
        self.folder = None
        self.nb_saved = 0

    def initView(self, data_dict):
        """
        Add all objects described in data_dict.

        :param data_dict: Dictionary containing the meshes data for each environment.
        :return:
        """
        # List containing all the meshes created for each environment
        meshes = []

        # Loop on the environments
        for idx in data_dict:
            # A model is the set of meshes in an environment
            model = data_dict[idx]
            self.models.append([])
            self.models_shapes[idx] = {}

            # Check if the field position is in model's data
            if 'positions' not in model:
                raise ValueError("[MeshVisualizer] Field 'positions' is missing to init the view.")
            # Check if the field position_shape is in model's data
            if 'position_shape' not in model:
                raise ValueError("[MeshVisualizer] Field 'position_shape' is missing to init the view.")
            # Reshape positions, store the position_shape for the model
            positions_shape = array(model['position_shape'], dtype=int)
            positions = model['positions'].reshape(positions_shape)
            self.models_shapes[idx]['positions'] = positions_shape
            # All the fields must have the same number of mesh

            # Check if the field cells is in model's data
            if 'cells' not in model:
                cells = [[] for _ in range(len(positions))]
                print("[MeshVisualizer] Warning: As field 'cells' is missing to init the view, the mesh will only be a "
                      "pointcloud.")
            else:
                # Check if the field cell_shape is in model's data
                if 'cell_shape' not in model:
                    raise ValueError("[MeshVisualizer] Field 'cell_shape' is missing to init the view.")
                cells = model['cells'].reshape(array(model['cell_shape'], dtype=int))
                # Check that their is a cell array for each position array
                if len(cells) != len(positions):
                    raise ValueError("[MeshVisualizer] The number of cell array mismatch the number of position array.")

            # Set the window in with the model will be rendered. If 'at' is not a field of model then set to another one
            at = array(model['at'], dtype=int) if 'at' in model else [MAX_INT for _ in range(len(positions))]
            # Check that their is a at value for each position array
            if len(at) != len(positions):
                raise ValueError("[MeshVisualizer] The number of 'at' value mismatch the number of position array.")

            # Set the scalar field
            # Todo: not tested yet
            field_dict = model['field_dict'] if 'field_dict' in model else [{'scalar_field': None} for _ in range(len(positions))]

            # Create a vedo mesh
            for i in range(len(positions)):
                vedo_mesh = self.addObject(positions=positions[i], cells=cells[i], at=at[i], field_dict=field_dict[i])
                meshes.append(vedo_mesh)
                self.models[-1].append(vedo_mesh)

        self.viewer = Plotter(N=self.nb_view, title=self.params['title'], axes=self.params['axes'],
                                   sharecam=True, interactive=self.params['interactive'])
        for mesh in meshes:
            self.viewer.add(mesh, at=self.data[mesh]['at'])
        print(f"[MeshVisualizer] Set visualizer camera position then close visualizer window to start.")
        self.viewer.show(interactive=True)
        self.render()

    def addObject(self, positions, cells=None, at=MAX_INT, field_dict={'scalar_field': None}):
        """
       Add an object to vedo visualizer. If cells is None then it's a point cloud, otherwise it correspond
       to the object surface topology.

       :param numpy.ndarray positions: Array of shape [n,3] describing the positions of the point cloud
       :param cells: Array which contains the topology of the object.
       :param int at: Target renderer in which to render the object
       :param dict field_dict: Dictionary of format {'data_name':data} that is attached to the Mesh object

       :return:
       """
        # Create a mesh witht he given data. vedo generate a points cloud if cells is None
        mesh = Mesh([positions, cells])
        # mesh.property.SetPointSize(10)
        # Efficient way to look for key existence in a dict
        if 'scalar_field' in field_dict and field_dict['scalar_field'] is not None:
            if 'color_map' not in field_dict or field_dict['color_map'] is None:
                field_dict['color_map'] = self.colormap
            mesh.cmap(field_dict['color_map'], field_dict['scalar_field'])

        # Attach each know fields to the Mesh object
        self.data[mesh] = {'positions': positions, 'position_shape': array(positions.shape, dtype=int),
                           'cells': cells, 'at': self.addView(at)}
        for data_field in field_dict.keys():
            self.data[mesh][data_field] = field_dict[data_field]
        return mesh

    def addView(self, at):
        """
        Add a view to the plotter window

        :param int at: Index of the view to add.

        :return: The new view index
        """
        # First addView returns 0
        if at >= self.nb_view:
            at = self.nb_view
            self.nb_view += 1
        return at

    def render(self):
        """
        Render the meshes in the desired windows.

        :return:
        """
        # if self.viewer is None:
        #     self.viewer = Plotter(N=self.nb_view,
        #                                title=self.params['title'],
        #                                axes=self.params['axes'],
        #                                sharecam=False,
        #                                interactive=self.params['interactive'])
        #     for mesh in self.data:
        #         self.viewer.add(mesh, at=self.data[mesh]['at'])
        self.update()
        self.viewer.render()
        self.viewer.allowInteraction()

    def update(self, position=True, scalar_field=True, cells=False):
        """

        :param position:
        :param scalar_field:
        :param cells:
        :return:
        """
        # In case, to debug
        for model in self.data:
            if position and self.data[model]['positions'] is not None:
                shape = array(self.data[model]['position_shape'], dtype=int)
                model.points(self.data[model]['positions'].reshape(shape))
            if scalar_field and 'scalar_field' in self.data[model] and self.data[model]['scalar_field'] is not None:
                model.cmap(self.data[model]['color_map'], self.data[model]['scalar_field'])

    def updateFromBatch(self, batch):
        """
        update vedo environment using batch information

        :param batch: dict templated as
                        {0: {client_parameters},
                         1: {client_parameters},
                         ...
                         N-1: {client_parameters},
                         'in': input_learning data,
                         'out': output_learning data}

                      client_parameter contain all data sent by the environment to the environment manager (position, velocity, strain, etc...)
        :return:
        """
        # assume client order == vedo mesh order
        for idx, mesh in enumerate(self.data):
            if idx in batch:
                for key in self.data[mesh]:
                    if key in batch[idx]:
                        print(f"Field {key} updated")
                        self.data[mesh][key] = batch[idx][key]
        self.render()

    def updateFromSample(self, sample, idx):
        """
        Update vedo meshes using a sample from an environment.

        :param sample: Dict templated as {'position': [position_mesh_1, ..., position_mesh_N]}
        :param idx: Index of environment
        :return:
        """
        # Get the list of meshes recorded in vedo for the environment n°idx
        model = self.models[idx]
        # Loop on sample.keys()
        for key in sample:
            # Check that the key is used to update a mesh
            if key not in self.models_shapes[idx]:
                raise ValueError(f"[MeshVisualizer] The field {key} is not part of a mesh data.")
            # Reshape the array according to the number of meshes and the field's vector size
            sample[key] = sample[key].reshape(self.models_shapes[idx][key])
            # Check that the field contains an array for each mesh
            if len(sample[key]) != len(model):
                raise ValueError(f"[MeshVisualizer] The number of value in received data does not match the number of"
                                 f"meshes for field {key}.")
            # Update the field for each mesh
            for i in range(len(model)):
                self.data[model[i]][key] = sample[key][i]
        # Update view
        self.render()

    def saveSample(self, session_dir):
        """
        Save the samples as a .npz file

        :param str session_dir: Directory in which to save the file

        :return:
        """
        if self.folder is None:
            self.folder = osPathJoin(session_dir, 'stats/wrong_samples')
            makedirs(self.folder)
            from DeepPhysX_Core.utils import wrong_samples
            import shutil
            shutil.copy(wrong_samples.__file__, self.folder)
        filename = osPathJoin(self.folder, f'wrong_sample_{self.nb_saved}.npz')
        self.nb_saved += 1
        self.viewer.export(filename=filename)

    def createObjectData(self, data_dict, positions, cells, at=None):
        """
        Used from the environment to template the data dict to send.

        :param data_dict: Empty dictionary or return of a previous call to this method
        :param positions: Array which contains the coordinates of the mesh
        :param cells: Array which contains the topology of the mesh
        :param at: Index of window in which render the object
        :return:
        """
        # Convert positions to float and concatenate if the field exists
        positions = array([positions], dtype=float)
        data_dict['positions'] = positions if 'positions' not in data_dict \
            else concatenate((data_dict['positions'], positions))
        # Store the position shape as the array will be flatten when sent to TcpIpServer
        data_dict['position_shape'] = array(data_dict['positions'].shape, dtype=float)

        # Convert cells to float and concatenate if the field exists
        cells = array([cells], dtype=float)
        data_dict['cells'] = cells if 'cells' not in data_dict \
            else concatenate((data_dict['cells'], cells))
        # Store the cell shape as the array will be flatten when sent to TcpIpServer
        data_dict['cell_shape'] = array(data_dict['cells'].shape, dtype=float)

        # Convert 'at' to float and concatenate if the field exists
        if at is not None:
            at = array([at], dtype=float)
            data_dict['at'] = at if 'at' not in data_dict else concatenate((data_dict['at'], at))

        return data_dict

    def updateObjectData(self, data_dict, positions):
        """
        Used from the environment to template the updated data dict to send.

        :param data_dict: Empty dictionary or return of a previous call to this method
        :param positions: Array which contains the updated coordinates of the mesh
        :return:
        """
        # Convert positions to float and concatenate if the field exists
        positions = array([positions], dtype=float)
        data_dict['positions'] = positions if 'positions' not in data_dict \
            else concatenate((data_dict['positions'], positions))
        return data_dict
