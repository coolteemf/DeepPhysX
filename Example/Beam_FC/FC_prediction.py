import os
from torch.optim import Adam
from torch.nn import MSELoss

from Example.Beam_Env.BeamEnvironmentConfig import BeamEnvironmentConfig
from DeepPhysX_PyTorch.FC.FCConfig import FCConfig
from DeepPhysX.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX.Runner.BaseRunner import BaseRunner
from DeepPhysX_Sofa.Runner.SofaBaseRunner import SofaRunner

# ENVIRONMENT PARAMETERS
grid_resolution = [40, 10, 10]
grid_min = [0., 0., 0.]
grid_max = [100., 25., 25.]
fixed_box = [0., 0., 0., 0., 25, 25]
p_grid = {'grid_resolution': grid_resolution,
          'min': grid_min,
          'max': grid_max,
          'fixed_box': fixed_box}

# NETWORK PARAMETERS
nb_hidden_layers = 2
nb_dof = grid_resolution[0] * grid_resolution[1] * grid_resolution[2]
layers_dim = [nb_dof * 3] + [nb_dof * 3 for _ in range(nb_hidden_layers + 1)] + [nb_dof * 3]


def createScene(root_node=None, runSofa=True):
    # Environment config
    env_config = BeamEnvironmentConfig(root_node=root_node, p_grid=p_grid,
                                       always_create_data=True)

    # Network config
    net_config = FCConfig(network_name="beam_FC", save_each_epoch=True,
                          loss=MSELoss, lr=1e-5, optimizer=Adam,
                          dim_output=3, dim_layers=layers_dim)

    # Dataset config
    dataset_config = BaseDatasetConfig()
    BaseDatasetConfig(partition_size=1, generate_data=True, shuffle_dataset=True)

    man_dir = os.path.dirname(os.path.realpath(__file__)) + '/beam_FC_trained'
    runner = BaseRunner(session_name="beam_FC_prediction", dataset_config=dataset_config,
                        environment_config=env_config, network_config=net_config, manager_dir=man_dir, nb_samples=0)
    root_node = runner.manager.environmentManager.environment.root
    root_node.addObject(SofaRunner(runner=runner))
    return root_node


if __name__ == '__main__':
    root = createScene(runSofa=False)
    import Sofa.Gui
    Sofa.Gui.GUIManager.Init("main", "qglviewer")
    Sofa.Gui.GUIManager.createGUI(root, __file__)
    Sofa.Gui.GUIManager.SetDimension(1080, 1080)
    Sofa.Gui.GUIManager.MainLoop(root)
    Sofa.Gui.GUIManager.closeGUI()
