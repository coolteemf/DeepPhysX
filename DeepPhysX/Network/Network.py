class Network:

    def __init__(self, network_name, network_type):
        self.type = network_type
        self.name = network_name
        self.description = ""

    def forward(self, x):
        raise NotImplementedError

    def backward(self, gradient):
        pass

    def setDevice(self):
        pass

    def loadParameters(self, path):
        pass

    def getParameters(self):
        pass

    def saveParameters(self, path):
        pass

    def nbParameters(self):
        pass

    def getDescription(self):
        if len(self.description) == 0:
            self.description += "\nBASE NETWORK\n"
            self.description += "   Name: {}\n".format(self.name)
            self.description += "   Type: {}\n".format(self.type)
        return self.description
