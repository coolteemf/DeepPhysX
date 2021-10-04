from torch import device, set_num_threads, load, save, as_tensor
from torch import float as tfloat
from torch.cuda import is_available, empty_cache
from torch.nn import Module
from gc import collect
from psutil import cpu_count

from DeepPhysX_Core.Network.BaseNetwork import BaseNetwork


class TorchNetwork(Module, BaseNetwork):

    def __init__(self, config):
        Module.__init__(self)
        BaseNetwork.__init__(self, config)

    def forward(self, x):
        raise NotImplementedError

    def setTrain(self):
        self.train()

    def setEval(self):
        self.eval()

    def setDevice(self):
        if is_available():
            self.device = device('cuda')
            # Garbage collector run
            collect()
            empty_cache()
        else:
            self.device = device('cpu')
            set_num_threads(cpu_count(logical=True) - 1)
        self.to(self.device)
        print("[{}]: Device is {}".format(self.name, self.device))

    def loadParameters(self, path):
        self.load_state_dict(load(path, map_location=self.device))

    def getParameters(self):
        return self.state_dict()

    def saveParameters(self, path):
        path = path + '.pth'
        save(self.state_dict(), path)

    def nbParameters(self):
        return sum(p.numel() for p in self.parameters())

    def transformFromNumpy(self, x):
        x = as_tensor(x, dtype=tfloat, device=self.device)
        x.requires_grad_()
        return x

    def transformToNumpy(self, x):
        return x.cpu().detach().numpy()

    def __str__(self):
        description = BaseNetwork.__str__(self)
        description += f"    Device: {self.device}\n"
        return description
