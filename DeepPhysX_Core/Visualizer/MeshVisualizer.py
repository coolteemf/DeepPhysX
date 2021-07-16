import vedo
import copy
import os

from DeepPhysX_Core.Visualizer.VedoVisualizer import VedoVisualizer


class MeshVisualizer(VedoVisualizer):

    def __init__(self, title='VedoVisualizer', interactive_window=False, show_axes=False,
                 min_color='yellow', max_color='red', range_color=10):
        self.data = {}
        self.viewer = None
        self.colormap = vedo.buildPalette(color1=min_color, color2=max_color, N=range_color, hsv=False)
        self.nb_view = 0
        self.params = {'title': title, 'interactive': interactive_window, 'axes': show_axes}
        # Wrong samples parameters
        self.folder = None
        self.nb_saved = 0

    def addPoints(self, positions, scalar=None, at=0):
        points = vedo.Points(positions)
        if scalar is not None:
            points.cmap(self.colormap, scalar)
        at = self.addView(at)
        self.data[points] = {'positions': positions, 'scalar': scalar, 'at': at}

    def addMesh(self, positions, cells, scalar=None, at=0):
        mesh = vedo.Mesh([positions, cells])
        if scalar is not None:
            mesh.cmap(self.colormap, scalar)
        at = self.addView(at)
        self.data[mesh] = {'positions': positions, 'scalar': scalar, 'at': at}

    def addView(self, at):
        if at >= self.nb_view + 1:
            at = self.nb_view + 1 if at > self.nb_view + 1 else at
            self.nb_view += 1
        return at

    def update(self):
        for model in self.data.keys():
            model.points(copy.copy(self.data[model]['positions']))
            if self.data[model]['scalar'] is not None:
                model.cmap(self.colormap, self.data[model]['scalar'])

    def render(self):
        if self.viewer is None:
            self.viewer = vedo.Plotter(title=self.params['title'], axes=self.params['axes'], N=self.nb_view + 1,
                                       interactive=self.params['interactive'])
            for model in self.data.keys():
                self.viewer.add(model, at=self.data[model]['at'])
        self.update()
        self.viewer.render()
        self.viewer.allowInteraction()

    def saveSample(self, session_dir):
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
