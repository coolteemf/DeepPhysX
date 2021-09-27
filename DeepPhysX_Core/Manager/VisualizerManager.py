from sys import maxsize as MAX_INT

from DeepPhysX_Core.Visualizer.VedoVisualizer import VedoVisualizer, VedoObject


class VisualizerManager:

    def __init__(self, data_manager=None, visual_object=VedoObject):
        """
        Handle the 3D representation of the data from a visualizer.
        Allows easy access to basic functionalities of the visualizer

        :param DataManager datamanager: DataManager that handles the VisualizerManager
        :param visualizer_class: The class of the desired visualizer
        """
        self.data_manager = data_manager

        self.visualizer = VedoVisualizer(visual_object=visual_object)

    def getDataManager(self):
        """

        :return: DataManager that handles the VisualizerManager
        """
        return self.data_manager

    def initView(self, data_dict):
        self.visualizer.init(data_dict)

    def updateFromSample(self, sample, id):
        self.visualizer.updateFromSample(sample, id)

    def render(self):
        self.visualizer.render()

    def saveSample(self, session_dir):
        """
        Save the samples as a filetype defined by the visualizer

        :param str session_dir: Directory in which to save the file

        :return:
        """
        self.visualizer.saveSample(session_dir=session_dir)

