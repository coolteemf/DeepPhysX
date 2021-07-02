"""
NNBeam.py
Neural network simulated beam with random deformations.
Launched with the python script '../beamPrediction.py'.
"""

import copy
import numpy as np
import random

from DeepPhysX_Sofa.Environment.SofaEnvironment import SofaEnvironment


# Inherit from SofaEnvironment which allow to implement and create a Sofa scene in the DeepPhysX_Core pipeline
class NNBeam(SofaEnvironment):

    def __init__(self, root_node, config, idx_instance=1, visualizer_class=None):
        super(NNBeam, self).__init__(root_node, config, idx_instance, visualizer_class)
        # Scene configuration
        self.config = config
        # Keep a track of the actual step number
        self.nb_steps = 0

    def create(self, config):
        """
        Create the Sofa scene graph. Automatically called by SofaEnvironmentConfig.
        :param config: Dataclass of SofaEnvironmentConfig objects, contains the custom parameters of the environment
        :return: None
        """
        # Get the scene parameters
        p_grid = config.p_grid
        g_res = p_grid['grid_resolution']
        self.nb_node = g_res[0] * g_res[1] * g_res[2]
        self.grid_size = config.p_grid['grid_min'] + config.p_grid['grid_max']

        # BEAM NN NODE
        self.root.addChild('beamNN')
        # "Fake" solver (zero newton iteration, only to compute forces)
        self.root.beamNN.addObject('LegacyStaticODESolver', name='StaticSolver', newton_iterations=0, printLog=False)
        # Grid topology of the beam
        self.grid = self.root.beamNN.addObject('RegularGridTopology', name='Grid', min=p_grid['grid_min'],
                                               max=p_grid['grid_max'], nx=g_res[0], ny=g_res[1], nz=g_res[2])
        self.MO = self.root.beamNN.addObject('MechanicalObject', src='@Grid', name='MO', template='Vec3d',
                                             showObject=False)
        self.root.beamNN.addObject('HexahedronSetTopologyContainer', name='HexaTopology', src='@Grid')
        self.root.beamNN.addObject('HexahedronSetGeometryAlgorithms', template='Vec3d')
        self.root.beamNN.addObject('HexahedronSetTopologyModifier')
        # Surface of the grid
        self.surface = self.root.beamNN.addObject('QuadSetTopologyContainer', name='QuadTopology', src='@HexaTopology')
        self.root.beamNN.addObject('QuadSetGeometryAlgorithms', template='Vec3d')
        self.root.beamNN.addObject('QuadSetTopologyModifier')
        self.root.beamNN.addObject('Hexa2QuadTopologicalMapping', input="@HexaTopology", output="@QuadTopology")
        # The following methods could require config because self.config is set after self.create() call
        # Behavior
        self.createBehavior(config)
        # Visual
        self.createVisual(config)
        # Collision
        self.createCollision(config)

    def createBehavior(self, config):
        """
        Create the behavior parts of the Sofa scene graph.
        :param config: Dataclass of SofaEnvironmentConfig objects, contains the custom parameters of the environment
        :return: None
        """
        # External forces applied on the surface
        self.box = self.root.beamNN.addObject('BoxROI', name='ForceBox', box=self.grid_size, drawBoxes=True,
                                              drawSize=1)
        self.CFF = self.root.beamNN.addObject('ConstantForceField', name='CFF', showArrowSize='0.',
                                              forces=[0 for _ in range(3 * self.nb_node)],
                                              indices=list(iter(range(self.nb_node))))

    def createVisual(self, config):
        """
        Create the visual parts of the Sofa scene graph.
        :param config: Dataclass of SofaEnvironmentConfig objects, contains the custom parameters of the environment
        :return: None
        """
        # Visual style of the scene
        self.root.addObject('VisualStyle', displayFlags="showVisualModels")
        # Visual model
        self.root.beamNN.addChild('visual')
        self.root.beamNN.visual.addObject('OglModel', name="oglModel", src='@../Grid', color='white')
        self.root.beamNN.visual.addObject('BarycentricMapping', name="BaryMap2", input='@../MO', output='@./')
        # file = os.pardir + '/Beam/Beam_really_smooth.obj'
        # self.root.beamNN.visual.addObject('MeshObjLoader', name="Mesh", filename=file, translation=[100, 12.5, 12.5])
        # self.root.beamNN.visual.addObject('OglModel', name="oglModel", src='@Mesh', color='white')

    def createCollision(self, config):
        """
        Create the collision parts of the Sofa scene graph.
        :param config: Dataclass of SofaEnvironmentConfig objects, contains the custom parameters of the environment
        :return: None
        """
        pass

    def onSimulationInitDoneEvent(self, event):
        """
        Called within the Sofa pipeline at the end of the scene graph initialisation.
        :param event: Sofa Event
        :return: None
        """
        # Get the data sizes
        self.input_size = self.MO.position.value.shape
        self.output_size = self.MO.position.value.shape
        # Get the indices of node on the surface
        self.idx_surface = self.surface.quads.value.reshape(-1)
        self.initVisualizer()

    def initVisualizer(self):
        # Visualizer
        if self.visualizer is not None:
            self.visualizer.addPoints(positions=self.MO.position.value)
            self.visualizer.addMesh(positions=self.MO.position.value, cells=self.surface.quads.value)

    def onAnimateBeginEvent(self, event):
        """
        Called within the Sofa pipeline at the beginning of the time step.
        :param event: Sofa Event
        :return: None
        """
        # Reset position
        self.MO.position.value = self.MO.rest_position.value
        # Create a random box ROI, select nodes of the surface
        indices = []
        while len(indices) == 0:  # We need the intersection between box and surface to be non empty
            x_min = random.randint(self.grid_size[0], self.grid_size[3] - 10)
            x_max = random.randint(x_min + 10, self.grid_size[3])
            y_min = random.randint(self.grid_size[1], self.grid_size[4] - 10)
            y_max = random.randint(y_min + 10, self.grid_size[4])
            z_min = random.randint(self.grid_size[2], self.grid_size[5] - 10)
            z_max = random.randint(z_min + 10, self.grid_size[5])
            # Set a new bounding box
            self.root.beamNN.removeObject(self.box)
            self.box = self.root.beamNN.addObject('BoxROI', name='ForceBox', drawBoxes=True, drawSize=1,
                                                  box=[x_min, y_min, z_min, x_max, y_max, z_max])
            self.box.init()
            # Get the intersection with the surface
            indices = list(self.box.indices.value)
            indices = list(set(indices).intersection(set(self.idx_surface)))
        # Create a random constant force field for the nodes in the bbox
        F = np.random.random(3) - np.random.random(3)
        F = (10 / np.linalg.norm(F)) * F
        # Set a new constant force field (variable number of indices)
        self.root.beamNN.removeObject(self.CFF)
        self.CFF = self.root.beamNN.addObject('ConstantForceField', name='CFF', showArrowSize='1',
                                              indices=indices, force=list(F))
        self.CFF.init()

    def onAnimateEndEvent(self, event):
        """
        Called within the Sofa pipeline at the end of the time step.
        :param event: Sofa Event
        :return: None
        """
        # Count the steps
        self.nb_steps += 1

    def computeInput(self):
        """
        Compute the input to be given to the network. Automatically called by EnvironmentManager.
        :return: None
        """
        # Compute the input force to give to the network
        self.input = copy.copy(self.MO.force.value)

    def computeOutput(self):
        """
        Compute the output to be given to the network. Automatically called by EnvironmentManager.
        :return: None
        """
        # Compute the output deformation to compare with the prediction of the network
        self.output = copy.copy(self.MO.position.value - self.MO.rest_position.value)

    def applyPrediction(self, prediction):
        """
        Apply the prediction of the network in the Sofa environment. Automatically called by EnvironmentManager.
        :return: None
        """
        # Add the displacement to the initial position
        U = prediction[0]
        self.MO.position.value = self.MO.rest_position.array() + U
        # Render
        self.renderVisualizer()
