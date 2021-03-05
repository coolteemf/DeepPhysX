from DeepPhysX.Network.BaseNetwork import BaseNetwork


class BaseNetworkConfig:

    def __init__(self, network_name="", network_type=""):
        self.Network = BaseNetwork
        self.config = network_name, network_type
        self.description = ""

    def create(self):
        return self.Network(*self.config)

    def getDescription(self):
        if len(self.description) == 0:
            self.description += "\nBASE NETWORK CONFIG\n"
            self.description += "   Network object: {}\n".format(self.Network)
            self.description += "   Config: {}\n".format(self.config)
        return self.description
