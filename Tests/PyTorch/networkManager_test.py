import os
import torch
import math
import random
import numpy as np

from MyNetwork import MyNetwork
from DeepPhysX_PyTorch.Network.TorchNetworkConfig import TorchNetworkConfig
from DeepPhysX.Manager.NetworkManager import NetworkManager


x = np.linspace(-math.pi, math.pi, 200)
y = np.sin(x)

def createData(batch_size=1):
    data = np.empty((batch_size, 1), dtype=np.float32)
    target = np.empty((batch_size, 1), dtype=np.float32)
    idx = np.random.randint(0, 200, batch_size)
    for i in range(batch_size):
        data[i] = x[idx[i]]
        target[i] = y[idx[i]]
    return data, target


def main():

    # Reference
    print("TRAINING WITHOUT MANAGER")
    net = MyNetwork()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    for epoch in range(500):
        data, target = createData(64)
        data = torch.from_numpy(data)
        target = torch.from_numpy(target)
        prediction = net(data)
        loss = criterion(prediction, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(loss.item())

    # Using NetworkManager
    print("\nTRAINING WITH MANAGER")
    training_config = TorchNetworkConfig(network_class=MyNetwork,
                                         network_name="myNetwork",
                                         network_type="MyNetwork",
                                         loss=torch.nn.MSELoss,
                                         lr=0.05,
                                         optimizer=torch.optim.SGD,
                                         network_dir=None,
                                         save_each_epoch=False)
    training_manager = NetworkManager(session_name='TestSession',
                                      network_config=training_config,
                                      session_dir=os.path.join(os.getcwd(), 'networkManager/train/'),
                                      train=True)
    for epoch in range(3000):
        data, target = createData(64)
        loss = training_manager.optimizeNetwork(data, target)
        if epoch % 300 == 0:
            print(loss.item())
            if training_manager.save_each_epoch:
                training_manager.saveNetwork()
    network_dir = training_manager.network_dir
    training_manager.close()

    # Using for prediction only
    print("\nPREDICTION WITH MANAGER")
    prediction_config = TorchNetworkConfig(network_class=MyNetwork,
                                           network_name="myNetwork",
                                           network_type="MyNetwork",
                                           loss=torch.nn.L1Loss,
                                           network_dir=network_dir,
                                           which_network=0)
    prediction_manager = NetworkManager(session_name='TestSession',
                                        network_config=prediction_config,
                                        session_dir=os.path.join(os.getcwd(), 'networkManager/predict/'),
                                        train=False)
    result_str = "Target: sin({:.2f})={:.2f} // Prediction: sin({:.2f})={:.2f} // Error: {}"
    for _ in range(10):
        data, target = createData()
        prediction = prediction_manager.getPrediction(data)
        loss = prediction_manager.computeLoss(prediction, target)
        print(result_str.format(data[0, 0], target[0, 0], data[0, 0], prediction[0, 0], loss.item()))

    return


if __name__ == '__main__':
    main()
