class VedoObject:

    def __init__(self, visualizer):
        self.name = self.__class__.__name__
        self.visualizer = visualizer

    def createObjects(self, scene):
        """

        :param scene: A scene is a set of objects.
        :return:
        """
        raise NotImplementedError

    def addView(self, at):
        """
        Add a view to the plotter window

        :param int at: Index of the view to add.
        :return: The new view index
        """
        # First addView returns 0
        if at >= self.visualizer.nb_view:
            at = self.visualizer.nb_view
            self.visualizer.nb_view += 1
        return at

    def updateObjects(self):
        """

        :return:
        """
        raise NotImplementedError
