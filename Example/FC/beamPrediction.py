import os
from torch.optim import Adam
from torch.nn import MSELoss

from DeepPhysX_PyTorch.FC.FCConfig import FCConfig
from DeepPhysX.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX.Pipelines.BaseRunner import BaseRunner
from DeepPhysX_Sofa.Runner.SofaBaseRunner import SofaRunner

import sys
from Example.Beam.BeamConfig import BeamConfig
if len(sys.argv) == 1 or sys.argv[1] == 'NNBeam':
    from Example.Beam.NNBeam import NNBeam as Beam
elif sys.argv[1] == 'NNBeamInteraction':
    from Example.Beam.NNBeamInteraction import NNBeamInteraction as Beam
elif sys.argv[1] == 'NNBeamCollision':
    from Example.Beam.NNBeamCollision import NNBeamCollision as Beam
elif sys.argv[1] == 'BothBeams':
    from Example.Beam.BothBeams import BothBeams as Beam
elif sys.argv[1] == 'BothBeamsInteraction':
    from Example.Beam.BothBeamsInteraction import BothBeamsInteraction as Beam
elif sys.argv[1] == 'NNBeamBall':
    from Example.Beam.NNBeamBall import NNBeamBall as Beam
else:
    print("Unknown Environment with name", sys.argv[1])
    quit(0)


# ENVIRONMENT PARAMETERS
grid_resolution = [40, 10, 10]
grid_min = [0., 0., 0.]
grid_max = [100., 25., 25.]
fixed_box = [0., 0., 0., 0., 25., 25.]
free_box = [49.5, -0.5, -0.5, 100.5, 25.5, 25.5]
all_box = [0., 0., 0., 100.5, 25.5, 25.5]
p_grid = {'grid_resolution': grid_resolution,
          'min': grid_min,
          'max': grid_max,
          'fixed_box': fixed_box,
          'free_box': free_box,
          'all_box': all_box}

# NETWORK PARAMETERS
nb_hidden_layers = 2
nb_dof = grid_resolution[0] * grid_resolution[1] * grid_resolution[2]
layers_dim = [nb_dof * 3] + [nb_dof * 3 for _ in range(nb_hidden_layers + 1)] + [nb_dof * 3]


def createScene(root_node=None):
    # Environment config
    env_config = BeamConfig(environment_class=Beam, root_node=root_node, p_grid=p_grid, always_create_data=True)
    # Network config
    net_config = FCConfig(network_name="beam_FC", save_each_epoch=True,
                          loss=MSELoss, lr=1e-5, optimizer=Adam,
                          dim_output=3, dim_layers=layers_dim)
    # Dataset config
    dataset_config = BaseDatasetConfig()
    BaseDatasetConfig(partition_size=1, shuffle_dataset=True)
    # Runner
    man_dir = os.path.dirname(os.path.realpath(__file__)) + '/beam_FC_prediction'
    runner = BaseRunner(session_name="beam_FC_prediction", dataset_config=dataset_config,
                        environment_config=env_config, network_config=net_config, session_dir=man_dir, nb_steps=0,
                        record_inputs=True, record_outputs=True)
    root_node = runner.manager.environment_manager.environment.root
    root_node.addObject(SofaRunner(runner=runner))
    return root_node, runner


if __name__ == '__main__':
    root, runner = createScene()
    import Sofa.Gui
    Sofa.Gui.GUIManager.Init("main", "qglviewer")
    Sofa.Gui.GUIManager.createGUI(root, __file__)
    Sofa.Gui.GUIManager.SetDimension(1080, 1080)
    Sofa.Gui.GUIManager.MainLoop(root)
    Sofa.Gui.GUIManager.closeGUI()
    runner.close()
