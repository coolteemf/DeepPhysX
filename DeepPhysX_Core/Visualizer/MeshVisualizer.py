import vedo
import os
import numpy as np
from sys import maxsize as MAX_INT

from DeepPhysX_Core.Visualizer.VedoVisualizer import VedoVisualizer


class MeshVisualizer(VedoVisualizer):

    def __init__(self, title='VedoVisualizer', interactive_window=False, show_axes=False,
                 min_color='yellow', max_color='red', range_color=10):
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
        self.viewer = None
        self.colormap = vedo.buildPalette(color1=min_color, color2=max_color, N=range_color, hsv=False)
        self.nb_view = 0
        self.params = {'title': title, 'interactive': interactive_window, 'axes': show_axes}
        # Wrong samples parameters
        self.folder = None
        self.nb_saved = 0

    def initView(self, data_dict):
        """
        Add all objects described in data_dict.

        :param data_dict:
        :return:
        """
        for idx in data_dict:
            model = data_dict[idx]
            if 'positions' in model:
                # Position
                positions = model['positions']
                if 'position_shape' in model:
                    position_shape = np.array(model['position_shape'], dtype=int)
                    positions = positions.reshape(position_shape)
                else:
                    raise ValueError('[MeshVisualizer] You need to add a "position_shape" field')
                # Cells
                cells = model['cells'] if 'cells' in model else None
                if cells is not None and 'cell_size' in model:
                    cell_size = np.array(model['cell_size'], dtype=int)
                    cells = cells.reshape(cell_size)
                else:
                    raise ValueError('[MeshVisualizer] You need to add a "cell_size" field')
                # Other
                at = model['at'] if 'at' in model else MAX_INT
                field_dict = model['field_dict'] if 'field_dict' in model else {'scalar_field': None}

                self.addObject(positions=positions, cells=cells, at=at, field_dict=field_dict)
        if self.viewer is not None:
            self.render()

    def addObject(self, positions, cells=None, at=MAX_INT, field_dict={'scalar_field': None}):
        """
       Add an object to vedo visualizer. If cells is None then it's a point cloud, otherwise it correspond
       to the object surface topology.

       :param numpy.ndarray positions: Array of shape [n,3] describing the positions of the point cloud
       :param numpy.ndarray cells: Array which contains the topology of the object.
       :param int at: Target renderer in which to render the object
       :param dict field_dict: Dictionary of format {'data_name':data} that is attached to the Mesh object

       :return:
       """
        # Create a mesh witht he given data. vedo generate a points cloud if cells is None
        mesh = vedo.Mesh([positions, cells])
        # mesh.property.SetPointSize(10)
        # Efficient way to look for key existence in a dict
        if 'scalar_field' in field_dict and field_dict['scalar_field'] is not None:
            if 'color_map' not in field_dict or field_dict['color_map'] is None:
                field_dict['color_map'] = self.colormap
            mesh.cmap(field_dict['color_map'], field_dict['scalar_field'])

        # Attach each know fields to the Mesh object
        self.data[mesh] = {'positions': positions, 'position_shape': np.array(positions.shape, dtype=int),
                           'cells': cells, 'at': self.addView(at)}
        for data_field in field_dict.keys():
            self.data[mesh][data_field] = field_dict[data_field]

        if self.viewer is None:
            self.viewer = vedo.Plotter(N=self.nb_view,
                                       title=self.params['title'],
                                       axes=self.params['axes'],
                                       sharecam=False,
                                       interactive=self.params['interactive'])

        self.viewer.add(mesh, at=self.data[mesh]['at'])
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
                shape = np.array(self.data[model]['position_shape'], dtype=int)
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

    def saveSample(self, session_dir):
        """
        Save the samples as a .npz file

        :param str session_dir: Directory in which to save the file

        :return:
        """
        if self.folder is None:
            self.folder = os.path.join(session_dir, 'stats/wrong_samples')
            os.makedirs(self.folder)
            from DeepPhysX_Core.utils import wrong_samples
            import shutil
            shutil.copy(wrong_samples.__file__, self.folder)
        filename = os.path.join(self.folder, f'wrong_sample_{self.nb_saved}.npz')
        self.nb_saved += 1
        self.viewer.export(filename=filename)
