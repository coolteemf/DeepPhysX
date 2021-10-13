# Basic python imports
import torch
from time import time

# DeepPhysX's PyTorch imports
from DeepPhysX_PyTorch.UNet.UNetConfig import UNetConfig
from DeepPhysX_PyTorch.UNet.UnetDataTransformation import UnetDataTransformation
from DeepPhysX_PyTorch.Network.TorchOptimization import TorchOptimization


def main():

    # UNet configuration
    unet_config = UNetConfig(optimization_class=TorchOptimization,              # Class which defines loss and optimizes network
                             data_transformation_class=UnetDataTransformation,  # Class which defines data transformations
                             network_dir=None,                                  # Directory of a trained network parameters
                             network_name="MyUnet",                             # Nickname of the network
                             which_network=0,                                   # Instance of the network in case several where saved during training
                             save_each_epoch=False,                             # Save network parameters at the end of each training epoch
                             loss=None,                                         # Loss function associated with the learning process
                             lr=1e-5,                                           # Leaning rate
                             optimizer=None,                                    # Optimizer associated with the learning process
                             input_size=(5, 10, 10),                            # Size of the input used to compute data transformations
                             nb_dims=3,                                         # Number of dimensions of the input
                             nb_input_channels=1,                               # Number of channels of the input
                             nb_first_layer_channels=128,                       # Number of channels after the first layer
                             nb_output_channels=3,                              # Number of channels of the output
                             nb_steps=3,                                        # Number of steps on each U side of the network
                             two_sublayers=True,                                # Define if each UNet layer is duplicated or not
                             border_mode='same',                                # Define padding value: 'valid'=0 / 'same'=1
                             skip_merge=False,                                  # Define if outputs must be merged on same levels of the U shape or not
                             data_scale=1.)                                     # Scale to apply for data transformations

    # Creating network, data_transformation and network_optimization
    # Those methods are automatically called by NetworkManager in a DeepPhysX pipeline
    unet = unet_config.createNetwork()
    unet.setDevice()
    data_transformation = unet_config.createDataTransformation()
    optimization = unet_config.createOptimization()
    print("NETWORK DESCRIPTION:", unet)
    print("DATA TRANSFORMATION DESCRIPTION:", data_transformation)
    print("OPTIMIZATION DESCRIPTION:", optimization)

    # Forward pass of Unet on a random tensor
    # Transformation methods are automatically called by NetworkManager in a DeepPhysX pipeline
    t = torch.rand((500, 1), dtype=torch.float, device=unet.device)
    start_time = time()
    t = data_transformation.transformBeforePrediction(t)
    pred = unet.forward(t)
    pred, _ = data_transformation.transformBeforeLoss(pred, None)
    pred = data_transformation.transformBeforeApply(pred)
    end_time = time()
    print(f"Prediction time: {round(end_time - start_time, 5) * 1e3} ms")
    print("Output shape:", pred.shape)


if __name__ == '__main__':
    main()
