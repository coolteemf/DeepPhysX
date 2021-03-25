import torch
from psutil import cpu_count

from DeepPhysX.Network.Network import Network


class PyTorchNetwork(torch.nn.Module, Network):

    def __init__(self, network_name, network_type):
        torch.nn.Module.__init__(self)
        Network.__init__(self, network_name, network_type)

    def setTrain(self):
        self.train()

    def setEval(self):
        self.eval()

    def setDevice(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            torch.set_num_threads(cpu_count(logical=True) - 1)

    def loadParameters(self, path):
        self.load_state_dict(torch.load(path))

    def getParameters(self):
        return self.state_dict()

    def saveParameters(self, path):
        torch.save(self.state_dict(), path)

    def nbParameters(self):
        return sum(p.numel() for p in self.parameters())
