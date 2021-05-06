import torch
from psutil import cpu_count

from DeepPhysX.Network.BaseNetwork import BaseNetwork


class PyTorchBaseNetwork(torch.nn.Module, BaseNetwork):

    def __init__(self, config):
        torch.nn.Module.__init__(self)
        BaseNetwork.__init__(self, config)
        self.descriptionName = "PYTORCH Network"

    def forward(self, x):
        raise NotImplementedError

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
        self.to(self.device)

    def loadParameters(self, path):
        self.load_state_dict(torch.load(path))

    def getParameters(self):
        return self.state_dict()

    def saveParameters(self, path):
        torch.save(self.state_dict(), path)

    def nbParameters(self):
        return sum(p.numel() for p in self.parameters())

    def transformFromNumpy(self, x):
        x = torch.from_numpy(x)
        return x

    def transformToNumpy(self, x):
        return x.detach().numpy()

    def transformInput(self):
        pass

    def transformPrediction(self):
        pass

    def transformGroundTruth(self):
        pass
