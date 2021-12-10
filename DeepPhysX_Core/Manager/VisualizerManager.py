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
        self.visualizer = visualizer()

    def getDataManager(self):
        """
        Return the manager that handles the VisualizerManager.

        :return: DataManager that handles the VisualizerManager
        """
        return self.data_manager

    def initView(self, data_dict):
        """
        Init the visualization window.

        :param data_dict: Dictionary containing all the visualization data fields.
        :return:
        """

        self.visualizer.initView(data_dict)

    def updateVisualizer(self, sample):
        self.visualizer.updateVisualizer(sample)

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

