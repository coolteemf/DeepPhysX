import os

from DeepPhysX.Network.Network import Network
from DeepPhysX.Network.NetworkConfig import NetworkConfig
from DeepPhysX.Manager.NetworkManager import NetworkManager


class MyNetwork(Network):

    def __init__(self, network_name, network_type):
        Network.__init__(self, network_name, network_type)


def main():

    network_config = NetworkConfig(network_class=MyNetwork,
                                   network_name="myNetwork",
                                   network_type="empty",
                                   loss=None,
                                   lr=None,
                                   optimizer=None,
                                   network_dir=None,
                                   save_each_epoch=False,
                                   which_network=1)
    network_manager = NetworkManager(session_name='TestSession',
                                     network_config=network_config,
                                     manager_dir=os.path.join(os.getcwd(), 'datasetManager/create/'),
                                     trainer=True)

    return


if __name__ == '__main__':
    main()
