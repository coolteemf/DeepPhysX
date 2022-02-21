"""
prediction_compare.py
Launch the prediction session in a SOFA GUI. Compare the two models.
"""

# Python imports
import os

# Sofa imports
import Sofa.Gui

# DeepPhysX Core imports
from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX_Sofa.Runner.SofaRunner import SofaRunner

# DeepPhysX Torch imports
from DeepPhysX_PyTorch.FC.FCConfig import FCConfig

# DeepPhysX Sofa imports
from DeepPhysX_Sofa.Environment.SofaEnvironmentConfig import SofaEnvironmentConfig

# Working session imports
from Environment.ArmadilloPrediction import ArmadilloPrediction
import Environment.parameters as parameters


def create_runner():
    """
    Launch DeepPhysX training session.

    :return:
    """

    # Environment config
    env_config = SofaEnvironmentConfig(environment_class=ArmadilloPrediction,
                                       as_tcp_ip_client=False)

    # UNet config
    nb_hidden_layers = 2
    nb_neurons = parameters.p_model.nb_nodes * 3
    layers_dim = [nb_neurons] + [nb_neurons for _ in range(nb_hidden_layers + 1)] + [nb_neurons]
    net_config = FCConfig(network_name='armadillo_FC',
                          dim_output=3,
                          dim_layers=layers_dim)

    # Dataset config
    dataset_config = BaseDatasetConfig(partition_size=1, shuffle_dataset=True)

    # Runner
    runner = SofaRunner(session_dir="sessions/armadillo",
                        dataset_config=dataset_config,
                        environment_config=env_config,
                        network_config=net_config,
                        nb_steps=0)
    return runner


if __name__ == '__main__':

    # Create SOFA runner
    runner = create_runner()

    # Launch SOFA GUI
    Sofa.Gui.GUIManager.Init("main", "qglviewer")
    Sofa.Gui.GUIManager.createGUI(runner.root, __file__)
    Sofa.Gui.GUIManager.SetDimension(1080, 1080)
    Sofa.Gui.GUIManager.MainLoop(runner.root)
    Sofa.Gui.GUIManager.closeGUI()

    # Manually close the runner (security if stuff like additional dataset need to be saved)
    runner.close()

    # Delete unwanted files
    for file in os.listdir(os.path.dirname(os.path.abspath(__file__))):
        if '.ini' in file or '.log' in file:
            os.remove(file)
