import vedo
import copy
import os
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

        # Efficient way to look for key existence in a dict
        if 'color_map' in field_dict and 'scalar_field' in field_dict \
                and field_dict['color_map'] is not None and field_dict['scalar_field'] is not None:
            mesh.cmap(field_dict['color_map'], field_dict['scalar_field'])
        elif 'scalar_field' in field_dict and field_dict['scalar_field'] is not None:
            mesh.cmap(self.colormap, field_dict['scalar_field'])

        # Attach each know fields to the Mesh object
        self.data[mesh] = {'positions': positions, 'at': self.addView(at)}
        for data_field in field_dict.keys():
            self.data[mesh][data_field] = field_dict[data_field]

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
        if self.viewer is None:
            self.viewer = vedo.Plotter(title=self.params['title'], axes=self.params['axes'], N=self.nb_view,
                                       interactive=self.params['interactive'])
            for model in self.data.keys():
                self.viewer.add(model, at=self.data[model]['at'])

        self.viewer.render()
        self.viewer.allowInteraction()

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
