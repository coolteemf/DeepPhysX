class Network:

    def __init__(self, network_name, network_type):
        self.type = network_type
        self.name = network_name
        self.device = None
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

    def getDescription(self):
        if len(self.description) == 0:
            self.description += "\nBASE NETWORK\n"
            self.description += "   Name: {}\n".format(self.name)
            self.description += "   Type: {}\n".format(self.type)
        return self.description
