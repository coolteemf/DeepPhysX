import os
import numpy as np
import functools

import Sofa
import Sofa.Gui
import SofaRuntime
import SofaCaribou

from DeepPhysX_Sofa.Environment.SofaEnvironment import SofaEnvironment, BytesNumpyConverter
from Sessions.Beam.utils import ndim_interpolation, amplitude

# Add the listed plugin to Sofa environment so we can run the scene
SofaRuntime.PluginRepository.addFirstPath(os.environ['CARIBOU_INSTALL'])
required_plugins = ['SofaComponentAll', 'SofaCaribou', 'SofaBaseTopology',
                    'SofaEngine', 'SofaBoundaryCondition', 'SofaTopologyMapping']

# ENVIRONMENT PARAMETERS
grid_params = {'grid_resolution': [40, 10, 10],  # Number of slices along each axis
               'grid_min': [0, 0, 0],  # Lowest point of the grid
               'grid_max': [100, 25, 25],  # Highest point of the grid
               'fixed_box': [0, 0, 0, 0, 25, 25]}  # Points withing this box will be fixed by Sofa

grid_node_count = functools.reduce(lambda a, b: a * b, grid_params['grid_resolution'])
grid_dofs_count = grid_node_count * 3
count = [16, 16, 16]

eigenVec = np.load(os.path.join(os.getcwd(), 'eigenVec.npy'))
mod_factors = 0.20
L = grid_params['grid_max'][0]
mod_coef = L ** 2 * np.array([mod_factors, mod_factors, mod_factors])
modal_amplitude = ndim_interpolation(-1. * mod_coef, 1. * mod_coef, count, ignored_dim=[], technic=amplitude)
np.take(modal_amplitude, np.random.rand(modal_amplitude.shape[0]).argsort(), axis=0, out=modal_amplitude)
inv_L = 1 / L


# Inherit from SofaEnvironment which allow to implement and create a Sofa scene in the DeepPhysX_Core pipeline
class FEMBeam(Sofa.Core.Controller):

    def __init__(self, root_node, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        root_node.addObject('RequiredPlugin', pluginName=required_plugins)
        self.root = root_node
        self.nb_step = 0
        self.last_U = None
        self.last_F = None
        self.last_I = None

    def create(self):
        """
        Create the Sofa scene graph. Automatically called by SofaEnvironmentConfig.
        :return: None
        """
        self.createFEM()
        self.createNN()

    def createFEM(self):
        # BeamFEM node
        self.root.addChild('beamFEM')

        # ODE solver + static solver
        self.root.beamFEM.addObject('LegacyStaticODESolver', name='ODESolver', newton_iterations=20,
                                    correction_tolerance_threshold=1e-8, residual_tolerance_threshold=1e-8,
                                    printLog=False)
        self.root.beamFEM.addObject('ConjugateGradientSolver', name='StaticSolver', preconditioning_method='Diagonal',
                                    maximum_number_of_iterations=1000, residual_tolerance_threshold=1e-8,
                                    printLog=False)

        # Grid topology of the beam
        self.root.beamFEM.addObject('RegularGridTopology', name='Grid', min=grid_params['grid_min'],
                                    max=grid_params['grid_max'], nx=grid_params['grid_resolution'][0],
                                    ny=grid_params['grid_resolution'][1], nz=grid_params['grid_resolution'][2])
        self.MO = self.root.beamFEM.addObject('MechanicalObject', src='@Grid', name='MO', template='Vec3d',
                                              showObject=False)

        # Volume topology of the grid
        self.root.beamFEM.addObject('HexahedronSetTopologyContainer', name='Hexa_Topology', src='@Grid')
        self.root.beamFEM.addObject('HexahedronSetGeometryAlgorithms', template='Vec3d')
        self.root.beamFEM.addObject('HexahedronSetTopologyModifier')

        # Surface topology of the grid
        self.surface = self.root.beamFEM.addObject('QuadSetTopologyContainer', name='Quad_Topology',
                                                   src='@Hexa_Topology')
        self.root.beamFEM.addObject('QuadSetGeometryAlgorithms', template='Vec3d')
        self.root.beamFEM.addObject('QuadSetTopologyModifier')
        self.root.beamFEM.addObject('Hexa2QuadTopologicalMapping', input="@Hexa_Topology", output="@Quad_Topology")

        # Constitutive law of the beam
        self.root.beamFEM.addObject('NeoHookeanMaterial', young_modulus=4500, poisson_ratio=0.45, name="NH")
        self.root.beamFEM.addObject('HyperelasticForcefield', material="@NH", template="Hexahedron",
                                    topology='@Hexa_Topology', printLog=True)

        # Fixed section of the beam
        self.root.beamFEM.addObject('BoxROI', box=grid_params['fixed_box'], name='Fixed_Box')
        self.fixed_cst = self.root.beamFEM.addObject('FixedConstraint', indices='@Fixed_Box.indices')

        # Forcefield through which the external forces are applied
        self.CFF = self.root.beamFEM.addObject('ConstantForceField', name='CFF', showArrowSize='0.1',
                                               forces=[0 for _ in range(grid_dofs_count)],
                                               indices=list(iter(range(grid_node_count))))

        # Adapt min and max of the grid
        grid_min = np.array(grid_params['grid_min']) - np.array([1, 1, 1])
        grid_max = np.array(grid_params['grid_max']) + np.array([1, 1, 1])

        # Create Spheres
        self.spheres_ROI = []
        for i in range(100):
            xyz = np.random.random(3)
            center = grid_min + xyz * (grid_max - grid_min)
            radius = 5 + 12.5 * np.random.random()
            self.spheres_ROI.append(
                self.root.beamFEM.addObject('SphereROI', template='Vec3d', centers=center.tolist(), radii=radius,
                                            drawSphere=False, drawSize=0.1))

        # Create Boxes
        self.boxes_ROI = []
        for i in range(100):
            xyz = np.random.random((3, 2))
            bmin = grid_min + np.amin(xyz, axis=1) * (grid_max - grid_min)
            bmax = grid_min + (0.5 * np.ones(3) + 0.5 * np.amax(xyz, axis=1)) * (grid_max - grid_min)
            box = bmin.tolist() + bmax.tolist()
            self.boxes_ROI.append(
                self.root.beamFEM.addObject('BoxROI', template='Vec3d', box=box, drawBoxes=False, drawSize=0.1))

    def createNN(self):
        # BeamNN node
        self.root.addChild('beamNN')

        # ODE solver + static solver
        self.root.beamNN.addObject('LegacyStaticODESolver', name='ODESolver', newton_iterations=20,
                                   correction_tolerance_threshold=1e-8, residual_tolerance_threshold=1e-8,
                                   printLog=True)
        self.root.beamNN.addObject('ConjugateGradientSolver', name='StaticSolver', preconditioning_method='Diagonal',
                                   maximum_number_of_iterations=1000, residual_tolerance_threshold=1e-9,
                                   printLog=False)

        # Grid topology of the beam
        self.root.beamNN.addObject('RegularGridTopology', name='Grid', min=grid_params['grid_min'],
                                   max=grid_params['grid_max'], nx=grid_params['grid_resolution'][0],
                                   ny=grid_params['grid_resolution'][1], nz=grid_params['grid_resolution'][2])
        self.NN_MO = self.root.beamNN.addObject('MechanicalObject', src='@Grid', name='MO', template='Vec3d',
                                                showObject=False)

        # Volume topology of the grid
        self.root.beamNN.addObject('HexahedronSetTopologyContainer', name='Hexa_Topology', src='@Grid')
        self.root.beamNN.addObject('HexahedronSetGeometryAlgorithms', template='Vec3d')
        self.root.beamNN.addObject('HexahedronSetTopologyModifier')

        # Surface topology of the grid
        self.root.beamNN.addObject('QuadSetTopologyContainer', name='Quad_Topology', src='@Hexa_Topology')
        self.root.beamNN.addObject('QuadSetGeometryAlgorithms', template='Vec3d')
        self.root.beamNN.addObject('QuadSetTopologyModifier')
        self.root.beamNN.addObject('Hexa2QuadTopologicalMapping', input="@Hexa_Topology", output="@Quad_Topology")

        # Constitutive law of the beam
        self.root.beamNN.addObject('NeoHookeanMaterial', young_modulus=4500, poisson_ratio=0.45, name="NH")
        self.root.beamNN.addObject('HyperelasticForcefield', material="@NH", template="Hexahedron",
                                   topology='@Hexa_Topology', printLog=True)

        # Fixed section of the beam
        self.root.beamNN.addObject('BoxROI', box=grid_params['fixed_box'], name='Fixed_Box')
        self.root.beamNN.addObject('FixedConstraint', indices='@Fixed_Box.indices')

        # Forcefield through which the external forces are applied
        self.NN_CFF = self.root.beamNN.addObject('ConstantForceField', name='CFF', showArrowSize='0.1',
                                                 forces=[0 for _ in range(grid_dofs_count)],
                                                 indices=list(iter(range(grid_node_count))))

    def onSimulationInitDoneEvent(self, event):
        """
        Called within the Sofa pipeline at the end of the scene graph initialisation.
        :param event: Sofa Event
        :return: None
        """
        # Get the network data sizes
        self.input_size = self.CFF.forces.value.shape
        self.output_size = self.MO.position.value.shape

        # Get indices of nodes on which forces could be applied
        idx = np.zeros(grid_node_count, dtype=np.bool)
        idx[self.surface.quads.value] = True
        if len(self.fixed_cst.indices.value) > 0:
            idx[self.fixed_cst.indices.value] = False
        self.valid_idx = np.argwhere(idx).reshape(-1)

        # Remove spheres, store ROI indices
        self.spheres_idx = []
        for sphere in self.spheres_ROI[::-1]:
            idx = list(set(sphere.indices.value) & set(self.valid_idx))
            if len(idx) > 0:
                self.spheres_idx.append(idx)
            self.root.beamFEM.removeObject(sphere)

        # Remove boxes, store ROI indices
        self.boxes_idx = []
        for box in self.boxes_ROI[::-1]:
            idx = list(set(box.indices.value) & set(self.valid_idx))
            if len(idx) > 0:
                self.boxes_idx.append(idx)
            self.root.beamFEM.removeObject(box)

        # Init force normalization coefficients
        q = np.zeros(grid_dofs_count)
        q[[0, 1, 2]] = mod_coef
        potential_forces = np.dot(eigenVec, q)
        self.F_normalization_coef = 1.0 / np.linalg.norm(potential_forces)
        self.F_inv_normalization_coef = 1.0 / self.F_normalization_coef

    def onAnimateBeginEvent(self, event):
        """
        Called within the Sofa pipeline at the beginning of the time step.
        :param event: Sofa Event
        :return: None
        """

        # Reset position
        self.MO.position.value = self.MO.rest_position.value
        # self.NN_MO.position.value = self.NN_MO.rest_position.value

        # Generate force
        q = np.zeros(grid_dofs_count)
        q[[0, 1, 2]] = modal_amplitude[self.nb_step]
        potential_forces = np.dot(eigenVec, q)
        # Select indices
        pick = np.random.randint(2)
        if pick == 0:
            selected_indices = self.spheres_idx[np.random.randint(len(self.spheres_idx))]
        elif pick == 1:
            selected_indices = self.boxes_idx[np.random.randint(len(self.boxes_idx))]
        else:
            nb_forces = np.min([1 + int(np.random.exponential(scale=1.2)), grid_node_count])
            selected_indices = self.valid_idx[np.random.uniform(0, len(self.valid_idx), nb_forces).astype(int)]
        # Fill in force vector
        cff_forces = np.zeros((len(selected_indices), 3))
        for k, i in enumerate(selected_indices):
            f_i = potential_forces.reshape(self.MO.position.shape)[i]
            cff_forces[k] = f_i
        potential_ampli = np.linalg.norm(potential_forces)
        ampli = potential_ampli / np.linalg.norm(cff_forces)

        # Update CFF
        self.CFF.forces.value = ampli * cff_forces
        self.CFF.indices.value = selected_indices
        self.CFF.showArrowSize.value = 1 / ampli

        # self.NN_CFF.forces.value = self.CFF.forces.value
        # self.NN_CFF.indices.value = self.CFF.indices.value

        if self.last_U is not None:
            self.NN_CFF.forces.value = self.last_F
            self.NN_CFF.indices.value = self.last_I
            if np.random.randint(0, 3) < 2:
                print("Use prediction")
                U = self.last_U + np.random.normal(0, 10e-6, self.last_U.shape)
                self.NN_MO.position.value = self.NN_MO.rest_position.value + U

    def onAnimateEndEvent(self, event):
        """
        Called within the Sofa pipeline at the end of the time step.
        :param event: Sofa Event
        :return: None
        """

        if self.last_U is not None:
            print("NN initial residual", self.root.beamNN.ODESolver.squared_initial_residual)
            print("NN residuals", self.root.beamNN.ODESolver.squared_residuals)
            print("F2 norm", np.linalg.norm(self.last_U) ** 2)
            # print("Residual loss", self.F_normalization_coef * np.sqrt(self.root.beamNN.ODESolver.squared_initial_residual))
            print("Residual loss", self.root.beamNN.ODESolver.squared_initial_residual / (np.linalg.norm(self.last_U) ** 2))
            print()
        print("FEM initial residual", self.root.beamFEM.ODESolver.squared_initial_residual)
        print("FEM residuals", self.root.beamFEM.ODESolver.squared_residuals)

        # residual_loss = self.F_normalization_coef * np.sqrt(self.root.beamNN.ODESolver.squared_initial_residual)

        self.nb_step = (self.nb_step + 1) % len(modal_amplitude)
        self.last_F = self.CFF.forces.value
        self.last_I = self.CFF.indices.value
        self.last_U = self.MO.position.value - self.MO.rest_position.value

    def checkSample(self, check_input=True, check_output=True):
        """
        Check if the sample can be added to the batch in TcpIpServer
        :param check_input:
        :param check_output:
        :return:
        """
        # Check if solver converged
        if not self.root.beamFEM.ODESolver.converged.value:
            Sofa.Simulation.reset(self.root)
        return self.root.beamFEM.ODESolver.converged.value

    def applyPrediction(self, prediction):
        """
        Apply the prediction of the network in the Sofa environment. Automatically called by EnvironmentManager.
        :return: None
        """
        pass

    def close(self):
        print(f"Closing Env n°{self.instance_id}")

    def __str__(self):
        return f"Environment n°{self.instance_id} with tensor {self.tensor}"


if __name__ == '__main__':
    root = Sofa.Core.Node()
    env = root.addObject(FEMBeam(root_node=root))
    env.create()
    Sofa.Simulation.init(root)

    # Launch the GUI
    Sofa.Gui.GUIManager.Init("main", "qglviewer")
    Sofa.Gui.GUIManager.createGUI(root, __file__)
    Sofa.Gui.GUIManager.SetDimension(1080, 1080)
    Sofa.Gui.GUIManager.MainLoop(root)
    Sofa.Gui.GUIManager.closeGUI()
