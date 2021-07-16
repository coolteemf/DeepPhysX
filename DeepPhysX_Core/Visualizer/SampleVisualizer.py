import os
import vedo

from DeepPhysX_Core.Visualizer.VedoVisualizer import VedoVisualizer


class SampleVisualizer(VedoVisualizer):

    def __init__(self, folder):
        super(SampleVisualizer, self).__init__()
        # Load samples in the folder
        files = sorted([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))
                        and f.endswith('.npz')])
        self.samples = [os.path.join(folder, f) for f in files]
        self.id_sample = 0
        # Create visualizer
        self.view = vedo.Plotter(title='SampleVisualizer', N=1, axes=0, interactive=True, offscreen=False)
        self.view.addButton(fnc=self.showPreviousSample, pos=(0.3, 0.005), states=["previous"])
        self.view.addButton(fnc=self.showNextSample, pos=(0.7, 0.005), states=["next"])
        # Load and show first sample
        self.current_sample = self.loadSample()
        self.view.show(self.current_sample)

    def showPreviousSample(self):
        if self.id_sample > 0:
            self.id_sample -= 1
            self.view.clear(self.current_sample)
            self.current_sample = self.loadSample()
            self.view.show(self.current_sample)

    def showNextSample(self):
        if self.id_sample < len(self.samples) - 1:
            self.id_sample += 1
            self.view.clear(self.current_sample)
            self.current_sample = self.loadSample()
            self.view.show(self.current_sample)

    def loadSample(self):
        filename = self.samples[self.id_sample]
        view = vedo.load(filename)
        return view.actors
