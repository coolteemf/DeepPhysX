import vedo
import copy
import os

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

    def addPoints(self, positions, scalar=None, at=0):
        """
        Add a point could to vedo visualizer

        :param numpy.ndarray positions: Array of shape [n,3] describing the positions of the point cloud
        :param numpy.ndarray scalar: Array of shape [n] valuing each point.
        :param int at: Target renderer in which to render the object

        :return:
        """
        points = vedo.Points(positions)
        if scalar is not None:
            points.cmap(self.colormap, scalar)
        at = self.addView(at)
        self.data[points] = {'positions': positions, 'scalar': scalar, 'at': at}

    def addMesh(self, positions, cells, scalar=None, at=0):
        """
       Add a point could to vedo visualizer

       :param numpy.ndarray positions: Array of shape [n,3] describing the positions of the point cloud
       :param numpy.ndarray cells: Array which contains the topology of the object.
       :param numpy.ndarray scalar: Array of shape [n] valuing each point.
       :param int at: Target renderer in which to render the object

       :return:
       """
        mesh = vedo.Mesh([positions, cells])
        if scalar is not None:
            mesh.cmap(self.colormap, scalar)
        at = self.addView(at)
        self.data[mesh] = {'positions': positions, 'scalar': scalar, 'at': at}

    def addView(self, at):
        """
        Add a view to the plotter window

        :param int at: Index of the view to add.

        :return:
        """
        if at >= self.nb_view + 1:
            self.nb_view += 1
        return self.nb_view

    def update(self):
        """
        Update the various field of a mesh for the vedo plotter

        :return:
        """
        for model in self.data.keys():
            model.points(copy.copy(self.data[model]['positions']))
            if self.data[model]['scalar'] is not None:
                model.cmap(self.colormap, self.data[model]['scalar'])

    def render(self):
        """
        Render the meshes in the desired windows.

        :return:
        """
        # TODO Add update queue for each modified field of a mesh
        if self.viewer is None:
            self.viewer = vedo.Plotter(title=self.params['title'], axes=self.params['axes'], N=self.nb_view + 1,
                                       interactive=self.params['interactive'])
            for model in self.data.keys():
                self.viewer.add(model, at=self.data[model]['at'])
        self.update()
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
        self.update()
        filename = os.path.join(self.folder, f'wrong_sample_{self.nb_saved}.npz')
        self.nb_saved += 1
        self.viewer.export(filename=filename)
