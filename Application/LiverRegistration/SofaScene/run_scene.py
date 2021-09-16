"""
run_scene.py
Run this script outside of any DPX pipeline, use Sofa to check the scene behaviour.
Run with: 'python3 liverSofa.py'
"""

import Sofa.Gui
from DeepPhysX_Core.Manager.EnvironmentManager import EnvironmentManager
from DeepPhysX_Sofa.Environment.SofaEnvironmentConfig import SofaEnvironmentConfig
from Application.LiverRegistration.SofaScene.Livers import Livers


def createScene():
    """
    Automatically called when launching a Sofa scene or called from main to create the scene graph.

    :return: root_node
    """

    # Set SofaEnvironment config
    env_config = SofaEnvironmentConfig(environment_class=Livers, always_create_data=True,
                                       as_tcpip_client=False)

    # Create Environment through EnvironmentManager
    env_manager = EnvironmentManager(environment_config=env_config)

    return env_manager.environment.root


if __name__ == '__main__':

    # Create scene graph
    root = createScene()

    # Launch Sofa GUI
    Sofa.Gui.GUIManager.Init("main", "qglviewer")
    Sofa.Gui.GUIManager.createGUI(root, __file__)
    Sofa.Gui.GUIManager.SetDimension(1080, 1080)
    Sofa.Gui.GUIManager.MainLoop(root)
    Sofa.Gui.GUIManager.closeGUI()
