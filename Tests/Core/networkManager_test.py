import os
import math
import numpy as np

from MyNetwork import MyBaseNetwork, MyBaseOptimisation
from DeepPhysX.Network.BaseNetworkConfig import BaseNetworkConfig
from DeepPhysX.Manager.NetworkManager import NetworkManager


def main():
    # Data
    inputs = np.linspace(-math.pi, math.pi, 2000)
    target = np.sin(inputs)

    # Reference
    print("TRAINING WITHOUT MANAGER")
    net = MyBaseNetwork()
    opt = MyBaseOptimisation(loss=None, lr=1e-6, optimizer=None)
    opt.setOptimizer(net)
    for t in range(2000):
        pred = net.forward(inputs)
        loss = opt.computeLoss(pred, target)
        opt.optimize(loss)
        if (t + 1) % 100 == 0:
            print(t + 1, loss['item'])
    print(f'Result: y = {net.a} + {net.b} x + {net.c} x^2 + {net.d} x^3')

    # Using networkManager
    print("\nTRAINING WITH MANAGER")
    training_config = BaseNetworkConfig(network_class=MyBaseNetwork,
                                        optimization_class=MyBaseOptimisation,
                                        network_name="myNetwork",
                                        lr=1e-6,
                                        network_dir=None,
                                        save_each_epoch=False)
    training_config.trainingMaterials = True
    training_manager = NetworkManager(session_name='TestSession',
                                      network_config=training_config,
                                      manager_dir=os.path.join(os.getcwd(), 'networkManager/train/'),
                                      trainer=True)
    for epoch in range(2000):
        loss = training_manager.optimizeNetwork(inputs, target)
        if (epoch + 1) % 100 == 0:
            print(epoch + 1, loss['item'])
            if training_manager.saveEachEpoch:
                training_manager.saveNetwork()
    network_dir = training_manager.networkDir
    training_manager.close()

    # Using for prediction only
    print("\nPREDICTION WITH MANAGER")
    prediction_config = BaseNetworkConfig(network_class=MyBaseNetwork,
                                          optimization_class=MyBaseOptimisation,
                                          network_name="myNetwork",
                                          network_dir=network_dir,
                                          which_network=0)
    prediction_manager = NetworkManager(session_name='TestSession',
                                        network_config=prediction_config,
                                        manager_dir=os.path.join(os.getcwd(), 'networkManager/predict/'),
                                        trainer=False)
    result_str = "Target: sin({:.2f})={:.2f} // Prediction: sin({:.2f})={:.2f} // Error: {}"
    idx = np.random.randint(0, 200, 10)
    for i in idx:
        prediction = prediction_manager.getPrediction(inputs[i])
        loss = prediction_manager.computeLoss(prediction, target)
        print(result_str.format(inputs[i], target[i], inputs[i], prediction, loss['item']))

    return


if __name__ == '__main__':
    main()
