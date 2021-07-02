import vedo

from DeepPhysX_Core.Visualizer.VedoVisualizer import VedoVisualizer


class MeshVisualizer(VedoVisualizer):

    def __init__(self, title='VedoVisualizer', interactive_window=False, show_axes=False):
        VedoVisualizer.__init__(self, title, interactive_window, show_axes)
        self.points = None
        self.mesh = None
        self.data = {}

    def addPoints(self, positions):
        self.points = vedo.Points(positions)
        self.view += self.points
        self.data[self.points] = positions

    def addMesh(self, positions, cells):
        self.mesh = vedo.Mesh([positions, cells])
        self.view += self.mesh
        self.data[self.mesh] = positions

    def update(self):
        for data in self.data.keys():
            data.points(self.data[data])

    def render(self):
        self.update()
        self.view.render()
        self.view.allowInteraction()
