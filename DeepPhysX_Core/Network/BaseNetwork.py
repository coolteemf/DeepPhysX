class BaseNetwork:

    def __init__(self, config):
        """
        BaseNetwork is a network class to compute predictions from input data according to actual state.

        :param config: BaseNetworkConfig.BaseNetwork.Properties class containing BaseNetwork parameters
        """

        self.name = self.__class__.__name__

        # Config
        self.device = None
        self.config = config

    def predict(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplementedError

    def setTrain(self):
        raise NotImplementedError

    def setEval(self):
        raise NotImplementedError

    def setDevice(self):
        raise NotImplementedError

    def loadParameters(self, path):
        raise NotImplementedError

    def getParameters(self):
        raise NotImplementedError

    def saveParameters(self, path):
        raise NotImplementedError

    def nbParameters(self):
        raise NotImplementedError

    def transformFromNumpy(self, x):
        raise NotImplementedError

    def transformToNumpy(self, x):
        raise NotImplementedError

    def __str__(self):
        """
        :return: String containing information about the BaseNetwork object
        """
        description = "\n"
        description += f"  {self.name}\n"
        description += f"    Name: {self.config.network_name}\n"
        description += f"    Type: {self.config.network_type}\n"
        description += f"    Number of parameters: {self.nbParameters()}\n"
        description += f"    Estimated size: {self.nbParameters() * 32 * 1.25e-10} Go\n"
        return description
