"""
FC.py
How to configure, create and use the Fully Connected Architecture.
"""

# Python related
import torch
from time import time

# DeepPhysX's PyTorch imports
from DeepPhysX_PyTorch.FC.FCConfig import FCConfig
from DeepPhysX_PyTorch.Network.TorchDataTransformation import TorchDataTransformation as TorchDT
from DeepPhysX_PyTorch.Network.TorchOptimization import TorchOptimization


def main():
    # FC configuration
    fc_config = FCConfig(optimization_class=TorchOptimization,  # Class which defines loss and optimizes network
                         data_transformation_class=TorchDT,     # Class which defines data transformations
                         network_dir=None,                      # Path with a trained network to load parameters
                         network_name="MyUnet",                 # Nickname of the network
                         which_network=0,                       # Instance index in case several where saved
                         save_each_epoch=False,                 # Save network parameters at each epoch end or not
                         lr=1e-5,                               # Leaning rate
                         require_training_stuff=True,           # Loss & optimizer are required or not for training
                         loss=torch.nn.MSELoss,                 # Loss class to use
                         optimizer=torch.optim.Adam,            # Optimizer class to manage the learning process
                         dim_output=3,                          # Number of dimensions of the output
                         dim_layers=[60, 60, 60, 60],           # Size of each layer of the FC
                         biases=[True, True, False])            # Layers contain biases or not

    """
    The following methods are automatically called by the NetworkManager in a normal DeepPhysX pipeline.
    They are only used here to demonstrate what is performed during the pipeline.
    """

    # Creating network, data_transformation and network_optimization
    print("Creating FC...")
    fc = fc_config.create_network()
    fc.set_device()
    fc.set_eval()
    data_transformation = fc_config.create_data_transformation()
    optimization = fc_config.create_optimization()
    print("\nNETWORK DESCRIPTION:", fc)
    print("\nDATA TRANSFORMATION DESCRIPTION:", data_transformation)
    print("\nOPTIMIZATION DESCRIPTION:", optimization)

    # Data transformations and forward pass of Unet on a random tensor
    t = torch.rand((1, 20, 3), dtype=torch.float, device=fc.device)
    start_time = time()
    fc_input = data_transformation.transform_before_prediction(t)
    fc_output = fc.forward(fc_input)
    fc_loss, _ = data_transformation.transform_before_loss(fc_output, None)
    fc_pred = data_transformation.transform_before_apply(fc_loss)
    fc_apply = fc_pred.reshape(t.shape)
    end_time = time()
    print(f"Prediction time: {round(end_time - start_time, 5) * 1e3} ms")
    print("Tensor shape:", t.shape)
    print("Input shape:", fc_input.shape)
    print("Output shape:", fc_output.shape)
    print("Loss shape:", fc_loss.shape)
    print("Prediction shape:", fc_pred.shape)
    print("Apply shape:", fc_apply.shape)


if __name__ == '__main__':
    main()
