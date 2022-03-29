"""
prediction.py
Launch the prediction session in a SOFA GUI with only predictions of the network.
"""

# Python related imports
import os

# Sofa related imports
import Sofa.Gui

# DeepPhysX related imports
from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX_Sofa.Pipeline.SofaRunner import SofaRunner
from DeepPhysX_PyTorch.FC.FCConfig import FCConfig
from DeepPhysX_Sofa.Environment.SofaEnvironmentConfig import SofaEnvironmentConfig

# Session related imports
from Environment.LiverPrediction import LiverPrediction
import Environment.parameters as parameters


def create_runner():

    # Environment config
    env_config = SofaEnvironmentConfig(environment_class=LiverPrediction,
                                       param_dict={'nb_forces': 2},
                                       as_tcp_ip_client=False)

    # UNet config
    nb_hidden_layers = 2
    nb_neurons = parameters.p_liver.nb_nodes * 3
    layers_dim = [nb_neurons] + [nb_neurons for _ in range(nb_hidden_layers + 1)] + [nb_neurons]
    net_config = FCConfig(network_name='armadillo_FC',
                          dim_output=3,
                          dim_layers=layers_dim,
                          biases=True)

    # Dataset config
    dataset_config = BaseDatasetConfig(normalize=True)

    # Define trained network session
    dpx_session, user_session = 'sessions/liver_training_dpx', 'sessions/liver_training_user_3'
    if not os.path.exists(dpx_session) and not os.path.exists(user_session):
        from download import download_training
        print('Downloading Demo trained network to launch prediction...')
        download_training()
    session_dir = user_session if os.path.exists(user_session) else dpx_session

    # Runner
    return SofaRunner(session_dir=session_dir,
                      dataset_config=dataset_config,
                      environment_config=env_config,
                      network_config=net_config,
                      nb_steps=0)


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
