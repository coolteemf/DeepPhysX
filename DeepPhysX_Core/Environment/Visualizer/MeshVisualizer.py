import vedo

from DeepPhysX_Core.Environment.Visualizer.VedoVisualizer import VedoVisualizer


class MeshVisualizer(VedoVisualizer):

    def __init__(self, title='VedoVisualizer', interactive_window=False, show_axes=False):
        super(MeshVisualizer).__init__(title, interactive_window, show_axes)
        self.points = None
        self.mesh = None

    def addPoints(self, positions):
        self.points = vedo.Points(positions)
        self.view += self.points

    def updatePoints(self, positions):
        if self.points is not None:
            self.points.points(positions)

    def addMesh(self, positions, cells):
        self.mesh = vedo.Mesh([positions, cells])
        self.view += self.mesh

    def updateMesh(self, positions):
        if self.mesh is not None:
            self.mesh.points(positions)
