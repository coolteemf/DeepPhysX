from sys import maxsize as MAX_INT


class VisualizerManager:

    def __init__(self,
                 data_manager=None,
                 visualizer_class=None):
        """
        Handle the 3D representation of the data from a visualizer.
        Allows easy access to basic functionalities of the visualizer

        :param DataManager datamanager: DataManager that handles the VisualizerManager
        :param visualizer_class: The class of the desired visualizer
        """
        self.data_manager = data_manager
        self.visualizer_class = visualizer_class
        self.visualizer = visualizer_class()
        pass

    def getDataManager(self):
        """

        :return: DataManager that handles the VisualizerManager
        """
        return self.data_manager

    def initVisualizer(self):
        """
        Init the visualizer with it's default parameters

        :return: The initialised visualizer
        """
        self.visualizer = self.visualizer_class()
        return self.visualizer

    def initVisualizerArgs(self):
        """
        Allows user to pass argument when initialising the visualizer

        Example :
            visu = visualizer_manager.initVisualizer()(name='Blue display')

        :return: The visualizer class
        """
        return self.visualizer_class

    def initView(self, data_dict):
        self.visualizer.initView(data_dict)

    def addObject(self, positions, cells=None, at=MAX_INT, field_dict={'scalar_field': None}):
        """
       Add an object tot he visualizer. If cells is None then it's a point cloud, otherwise it correspond
       to the object surface topology.

       :param numpy.ndarray positions: Array of shape [n,3] describing the positions of the point cloud
       :param numpy.ndarray cells: Array which contains the topology of the object.
       :param int at: Target renderer in which to render the object
       :param dict field_dict: Dictionary of format {'data_name':data} that is attached to the Mesh object

       :return:
       """
        return self.visualizer.addObject(positions=positions, cells=cells, at=at, field_dict=field_dict)

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

