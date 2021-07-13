import vedo
import copy


class MeshVisualizer:

    def __init__(self, title='VedoVisualizer', interactive_window=False, show_axes=False):
        self.data = {}
        self.viewer = None
        self.nb_view = 0
        self.params = {'title': title, 'interactive': interactive_window, 'axes': show_axes}

    def addPoints(self, positions, at=0):
        points = vedo.Points(positions)
        at = self.addView(at)
        self.data[points] = {'coords': positions, 'at': at}

    def addMesh(self, positions, cells, at=0):
        mesh = vedo.Mesh([positions, cells])
        at = self.addView(at)
        self.data[mesh] = {'coords': positions, 'at': at}

    def addView(self, at):
        if at >= self.nb_view + 1:
            at = self.nb_view + 1 if at > self.nb_view + 1 else at
            self.nb_view += 1
        return at

    def update(self):
        for model in self.data.keys():
            model.points(copy.copy(self.data[model]['coords']))

    def render(self):
        if self.viewer is None:
            self.viewer = vedo.Plotter(title=self.params['title'], axes=self.params['axes'], N=self.nb_view + 1,
                                       interactive=self.params['interactive'])
            for model in self.data.keys():
                self.viewer.add(model, at=self.data[model]['at'])
        self.update()
        self.viewer.render()
        self.viewer.allowInteraction()
