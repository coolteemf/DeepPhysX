

class BaseNetwork:

    def __init__(self, network_name, network_type):
        self.type = network_type
        self.name = network_name
        self.description = ""

    def setDevice(self):
        pass

    def loadParameters(self, path):
        """ Doc """
        pass

    def saveParameters(self, path):
        pass

    def getParameters(self):
        pass

    def nbParameters(self):
        return 0

    def getDescription(self):
        if len(self.description) == 0:
            self.description += "\nBASE NETWORK\n"
            self.description += "   Name: {}\n".format(self.name)
            self.description += "   Type: {}\n".format(self.type)
        return self.description

