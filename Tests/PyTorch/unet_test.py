from DeepPhysX_PyTorch.UNet.UNetConfig import UNetConfig
import torch

config = UNetConfig(network_name="Unet_test",
                    network_type="UNet",
                    loss=None,
                    lr=None,
                    optimizer=None,
                    network_dir=None,
                    save_each_epoch=False,
                    which_network=None,

                    steps=1,
                    first_layer_channels=1,
                    nb_classes=3,
                    nb_input_channels=1,
                    two_sublayers=True,
                    nb_dims=3,
                    border_mode='same')

unet = config.createNetwork()
print(unet)
print(unet.forward(torch.tensor([0.5])))
