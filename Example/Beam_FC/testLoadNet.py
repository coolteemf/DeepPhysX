import torch
torch.cuda.empty_cache()

from DeepPhysX_PyTorch.FC.FCConfig import FCConfig


grid_resolution = [40, 10, 10]
nb_hidden_layers = 2
nb_dof = grid_resolution[0] * grid_resolution[1] * grid_resolution[2]
layers_dim = [nb_dof * 3] + [nb_dof * 3 for _ in range(nb_hidden_layers + 1)] + [nb_dof * 3]


net_config = FCConfig(network_name="beam_FC", save_each_epoch=True,
                      loss=None, lr=1e-5, optimizer=None,
                      dim_output=3, dim_layers=layers_dim)
net = net_config.createNetwork()
net.setDevice()
print(net.device)
print(net)
del net



