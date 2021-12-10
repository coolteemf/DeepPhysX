from sys import maxsize as MAX_INT

from DeepPhysX_Core.Visualizer.NewVisualizer import NewVisualizer


class VisualizerManager:

    def __init__(self,
                 data_manager=None,
                 visualizer=None):
        """
        Handle the 3D representation of the data from a visualizer.
        Allows easy access to basic functionalities of the visualizer

        :param DataManager data_manager: DataManager that handles the VisualizerManager
        :param visual_object: The class of the desired vedo object
        """
        self.data_manager = data_manager
<<<<<<< HEAD
        self.visualizer = VedoVisualizer(visual_object=visual_object)
=======

        self.visualizer = visualizer()
>>>>>>> Changes from BaseEnvironment variable name

    def getDataManager(self):
        """
        Return the manager that handles the VisualizerManager.

        :return: DataManager that handles the VisualizerManager
        """
        return self.data_manager

    def initView(self, data_dict):
<<<<<<< HEAD
        """
        Init the visualization window.

        :param data_dict: Dictionary containing all the visualization data fields.
        :return:
        """
        self.visualizer.init(data_dict)
=======
        self.visualizer.initView(data_dict)
>>>>>>> Changes from BaseEnvironment variable name

    def updateFromSample(self, sample, index):
        """
        Update the rendering windows with a sample of visualization data.

        :param sample: Sample of updated visualization data
        :param index: ID of the client
        :return:
        """
        self.visualizer.updateFromSample(sample, index)

    def render(self):
        """
        Trigger a render step of the visualization window.

        :return:
        """
        self.visualizer.render()

    def saveSample(self, session_dir):
        """
        Save the samples as a filetype defined by the visualizer

        :param str session_dir: Directory in which to save the file

        :return:
        """
        self.visualizer.saveSample(session_dir=session_dir)

