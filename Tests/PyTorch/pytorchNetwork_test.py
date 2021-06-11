import torch

from DeepPhysX_PyTorch.Network.TorchNetwork import TorchNetwork
from DeepPhysX_PyTorch.Network.TorchNetworkConfig import TorchNetworkConfig


def main():

    # Network configuration
    network_config = TorchNetworkConfig(network_class=TorchNetwork,
                                        network_name="MyNetwork",
                                        network_type="Empty",
                                        loss=torch.nn.MSELoss,
                                        lr=1e-3,
                                        optimizer=torch.optim.SGD,
                                        network_dir=None,
                                        save_each_epoch=False,
                                        which_network=1)
    print(network_config.getDescription())

    # Network optimization
    network_optimization = network_config.createOptimization()
    print(network_optimization.getDescription())

    # Network
    network = network_config.createNetwork()
    print(network.getDescription())

    return


if __name__ == '__main__':
    main()
