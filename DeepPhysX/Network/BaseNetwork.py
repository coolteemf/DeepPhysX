class BaseNetwork:

    def __init__(self, config):
        self.type = config.network_type
        self.name = config.network_name
        self.device = None
        # Description
        self.descriptionName = "CORE Network"
        self.description = ""

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

    def getDescription(self):
        if len(self.description) == 0:
            self.description += "\n{}\n".format(self.descriptionName)
            self.description += "   Name: {}\n".format(self.name)
            self.description += "   Type: {}\n".format(self.type)
        return self.description
