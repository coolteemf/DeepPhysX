import torch
import gc
from psutil import cpu_count

from DeepPhysX_Core.Network.BaseNetwork import BaseNetwork


class TorchNetwork(torch.nn.Module, BaseNetwork):

    def __init__(self, config):
        torch.nn.Module.__init__(self)
        BaseNetwork.__init__(self, config)

    def forward(self, x):
        raise NotImplementedError

    def setTrain(self):
        self.train()

    def setEval(self):
        self.eval()

    def setDevice(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            gc.collect()
            torch.cuda.empty_cache()
        else:
            self.device = torch.device('cpu')
            torch.set_num_threads(cpu_count(logical=True) - 1)
        self.to(self.device)
        print("[{}]: Device is {}".format(self.name, self.device))

    def loadParameters(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))

    def getParameters(self):
        return self.state_dict()

    def saveParameters(self, path):
        path = path + '.pth'
        torch.save(self.state_dict(), path)

    def nbParameters(self):
        return sum(p.numel() for p in self.parameters())

    def transformFromNumpy(self, x):
        x = torch.as_tensor(x, dtype=torch.float, device=self.device)
        x.requires_grad_()
        return x

    def transformToNumpy(self, x):
        return x.cpu().detach().numpy()
