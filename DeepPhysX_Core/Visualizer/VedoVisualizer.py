import vedo

class VedoVisualizer:

    def __init__(self, title='VedoVisualizer', interactive_window=False, show_axes=False):
        self.view = vedo.Plotter(title=title, interactive=interactive_window, axes=show_axes)

    def render(self):
        self.view.render()
