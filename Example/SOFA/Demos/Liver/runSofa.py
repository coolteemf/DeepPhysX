"""
runSofa.py
Launch the LiverSofa Environment in a Sofa GUI.
"""

# Python related imports
import os

# Sofa related imports
import Sofa.Gui

# DeepPhysX related imports
from DeepPhysX_Core.Manager.EnvironmentManager import EnvironmentManager
from DeepPhysX_Sofa.Environment.SofaEnvironmentConfig import SofaEnvironmentConfig

# Working session imports
from Environment.LiverSofa import LiverSofa


def create_environment():

    # Create SofaEnvironment configuration
    environment_config = SofaEnvironmentConfig(environment_class=LiverSofa,
                                               as_tcp_ip_client=False)

    # Create Armadillo Environment within EnvironmentManager
    environment_manager = EnvironmentManager(environment_config=environment_config)
    return environment_manager.environment


if __name__ == '__main__':

    # Create Environment
    environment = create_environment()

    # Launch Sofa GUI
    Sofa.Gui.GUIManager.Init(program_name="main", gui_name="qglviewer")
    Sofa.Gui.GUIManager.createGUI(environment.root, __file__)
    Sofa.Gui.GUIManager.SetDimension(1080, 1080)
    Sofa.Gui.GUIManager.MainLoop(environment.root)
    Sofa.Gui.GUIManager.closeGUI()

    # Delete log files
    for file in os.listdir(os.getcwd()):
        if '.ini' in file or '.log' in file:
            os.remove(file)
