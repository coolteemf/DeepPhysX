"""
validation.py
Launch the prediction session in a SOFA GUI. Compare the two models.
Use 'python3 validation.py -e' to run the pipeline with newly created samples in Environment (default).
Use 'python3 validation.py -d' to run the pipeline with existing samples from a Dataset.
"""

# Python related imports
import os
import sys

# Sofa related imports
import Sofa.Gui

# DeepPhysX related imports
from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX_Sofa.Pipeline.SofaRunner import SofaRunner
from DeepPhysX_PyTorch.FC.FCConfig import FCConfig
from DeepPhysX_Sofa.Environment.SofaEnvironmentConfig import SofaEnvironmentConfig

# Session related imports
from Environment.LiverValidation import LiverValidation
import Environment.parameters as parameters


def create_runner(dataset_dir):

    # Environment config
    env_config = SofaEnvironmentConfig(environment_class=LiverValidation,
                                       param_dict={'compute_sample': dataset_dir is None},
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
    dataset_config = BaseDatasetConfig(shuffle_dataset=True,
                                       normalize=True,
                                       dataset_dir=dataset_dir,
                                       use_mode=None if dataset_dir is None else 'Validation')

    # Define trained network session
    dpx_session, user_session = 'sessions/liver_training_dpx', 'sessions/liver_training_user_3'
    if not os.path.exists(dpx_session) and not os.path.exists(user_session):
        from download import download_training
        print('Downloading Demo trained network to launch validation...')
        download_training()
    session_dir = user_session if os.path.exists(user_session) else dpx_session

    # Runner
    return SofaRunner(session_dir=session_dir,
                      dataset_config=dataset_config,
                      environment_config=env_config,
                      network_config=net_config,
                      nb_steps=500)


if __name__ == '__main__':

    # Get dataset option
    dataset = None
    if len(sys.argv) > 1:

        # Check script option
        if sys.argv[1] not in ['-e', '-d']:
            print("Script option must be either '-e' for samples produced in Environment (default) or '-d' for samples "
                  "loaded from an existing Dataset.")
            quit(0)

        # Check Dataset existence for normalization coefficients
        dpx_session = 'sessions/liver_data_dpx'
        # if not os.path.exists(dpx_session):
        #     from download import download_dataset
        #     print('Downloading Demo dataset to launch training...')
        #     download_dataset()

        dataset = dpx_session if sys.argv[1] == '-d' else None

    # Create SOFA runner
    runner = create_runner(dataset)

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
