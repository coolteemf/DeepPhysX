import sys

from Example.Beam.BeamConfig import BeamConfig

if len(sys.argv) == 1 or sys.argv[1] == 'FEMBeam':
    from Example.Beam.FEMBeam import FEMBeam as Beam
elif sys.argv[1] == 'FEMBeamInteraction':
    from Example.Beam.FEMBeamMouse import FEMBeamInteraction as Beam
else:
    raise ValueError("Unknown Environment with name", sys.argv[1])

# ENVIRONMENT PARAMETERS
grid_resolution = [26, 5, 5]
# grid_resolution = [40, 10, 10]
grid_min = [0., 0., 0.]
# grid_max = [100, 25, 25]
grid_max = [100, 15, 15]
fixed_box = [0., 0., 0., 0., 25, 25]
p_grid = {'grid_resolution': grid_resolution, 'min': grid_min, 'max': grid_max, 'fixed_box': fixed_box}


def createScene(root_node=None):
    # Environment config
    env_config = BeamConfig(environment_class=Beam, root_node=root_node, p_grid=p_grid, always_create_data=True)
    env = env_config.createEnvironment()
    root_node = env.root
    return root_node


if __name__ == '__main__':
    root = createScene()
    import Sofa.Gui
    Sofa.Gui.GUIManager.Init("main", "qglviewer")
    Sofa.Gui.GUIManager.createGUI(root, __file__)
    Sofa.Gui.GUIManager.SetDimension(1080, 1080)
    Sofa.Gui.GUIManager.MainLoop(root)
    Sofa.Gui.GUIManager.closeGUI()
