import torch
import time

from DeepPhysX_PyTorch.UNet.UNetConfig import UNetConfig
from DeepPhysX_PyTorch.UNet.UnetDataTransformation import UnetDataTransformation
from DeepPhysX_PyTorch.Network.TorchOptimization import TorchOptimization


def main():

    # UNet configuration
    unet_config = UNetConfig(optimization_class=TorchOptimization,
                             data_transformation_class=UnetDataTransformation,
                             network_dir=None,
                             network_name="MyUnet",
                             which_network=0,
                             save_each_epoch=False,
                             loss=None,
                             lr=1e-5,
                             optimizer=None,
                             input_size=(5, 10, 10),
                             nb_dims=3,
                             nb_input_channels=1,
                             nb_first_layer_channels=128,
                             nb_output_channels=3,
                             nb_steps=3,
                             two_sublayers=True,
                             border_mode='same',
                             skip_merge=False,
                             data_scale=1.)

    # Creating network, data_transformation and network_optimization
    # Those methods are automatically called by NetworkManager in a DeepPhysX pipeline
    unet = unet_config.createNetwork()
    unet.setDevice()
    data_transformation = unet_config.createDataTransformation()
    optimization = unet_config.createOptimization()
    # print("NETWORK DESCRIPTION:", unet)
    # print("DATA TRANSFORMATION DESCRIPTION:", data_transformation)
    # print("OPTIMIZATION DESCRIPTION:", optimization)

    # Forward pas of Unet on a random tensor
    t = torch.rand((500, 1), dtype=torch.float, device=unet.device)

    start_time = time.time()
    t = data_transformation.transformBeforePrediction(t)
    pred = unet.forward(t)
    pred, _ = data_transformation.transformBeforeLoss(pred, None)
    pred = data_transformation.transformBeforeApply(pred)
    end_time = time.time()
    print("Prediction time:", end_time - start_time)

if __name__ == '__main__':
    main()
