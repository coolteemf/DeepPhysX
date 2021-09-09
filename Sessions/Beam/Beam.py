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
class FEMBeam(SofaEnvironment):

    def __init__(self, root_node, ip_address='localhost', port=10000, data_converter=BytesNumpyConverter,
                 instance_id=1):
        super(FEMBeam, self).__init__(ip_address=ip_address, port=port, data_converter=data_converter,
                                      instance_id=instance_id, root_node=root_node)

        root_node.addObject('RequiredPlugin', pluginName=required_plugins)
        self.root = root_node
        self.nb_step = 0
        self.dataset_input_sample = None
        self.dataset_output_sample = None
        self.compute_fem_solution = True

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
                                    correction_tolerance_threshold=1e-9, residual_tolerance_threshold=1e-9,
                                    printLog=False)
        self.root.beamFEM.addObject('ConjugateGradientSolver', name='StaticSolver', preconditioning_method='Diagonal',
                                    maximum_number_of_iterations=1000, residual_tolerance_threshold=1e-9,
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
                                            drawSphere=False, drawSize=0.))

        # Create Boxes
        self.boxes_ROI = []
        for i in range(100):
            xyz = np.random.random((3, 2))
            bmin = grid_min + np.amin(xyz, axis=1) * (grid_max - grid_min)
            bmax = grid_min + (0.5 * np.ones(3) + 0.5 * np.amax(xyz, axis=1)) * (grid_max - grid_min)
            box = bmin.tolist() + bmax.tolist()
            self.boxes_ROI.append(
                self.root.beamFEM.addObject('BoxROI', template='Vec3d', box=box, drawBoxes=True, drawSize=0.1))

    def createNN(self):
        # BeamNN node
        self.root.addChild('beamNN')

        # ODE solver + static solver
        self.root.beamNN.addObject('LegacyStaticODESolver', name='ODESolver', newton_iterations=0,
                                   correction_tolerance_threshold=1e-6, residual_tolerance_threshold=1e-6,
                                   printLog=False)
        self.root.beamNN.addObject('ConjugateGradientSolver', name='StaticSolver', preconditioning_method='Diagonal',
                                   maximum_number_of_iterations=0, residual_tolerance_threshold=1e-9,
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

    def send_parameters(self):
        """
        Call by TcpIpClient when init is done to send data to TciIpServer
        :return:
        """
        # Send mesh data dict
        positions = np.array(self.MO.position.value, dtype=float)
        position_shape = np.array(positions.shape, dtype=float)  # Array is flatten when send to server, used to reshape
        cells = np.array(self.surface.quads.value, dtype=float)
        cell_size = np.array(cells.shape, dtype=float)  # Same here
        return {'positions': positions, 'position_shape': position_shape, 'cells': cells, 'cell_size': cell_size}

    def onAnimateBeginEvent(self, event):
        """
        Called within the Sofa pipeline at the beginning of the time step.
        :param event: Sofa Event
        :return: None
        """

        # Reset position
        self.MO.position.value = self.MO.rest_position.value
        self.NN_MO.position.value = self.NN_MO.rest_position.value

        # Generate force
        if self.compute_fem_solution:
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
            net_forces = np.zeros(self.MO.position.shape)
            for k, i in enumerate(selected_indices):
                f_i = potential_forces.reshape(self.MO.position.shape)[i]
                cff_forces[k] = f_i
                net_forces[i] = f_i
            potential_ampli = np.linalg.norm(potential_forces)
            ampli = potential_ampli / np.linalg.norm(net_forces.reshape(-1))
            # Update CFF
            self.initial_force = ampli * net_forces
            self.CFF.forces.value = ampli * cff_forces
            self.CFF.indices.value = selected_indices
            self.CFF.showArrowSize.value = 1 / ampli
        else:
            # Set CFF force value to 0
            self.CFF.force.value = np.zeros(3)
            # Get the dataset force vector, fill in NN_CFF force vector with non zero elements
            net_forces = self.dataset_input_sample
            ampli = 1.
            cff_forces = []
            selected_indices = []
            for i, f_i in enumerate(net_forces):
                if np.linalg.norm(f_i) != 0:
                    cff_forces.append(f_i * self.F_inv_normalization_coef)
                    selected_indices.append(i)

        # Set NN_CFF force values and indices
        self.NN_CFF.forces.value = ampli * cff_forces
        self.NN_CFF.indices.value = selected_indices

        # Request network prediction, update NN beam positions
        net_input = self.F_normalization_coef * self.initial_force if self.compute_fem_solution else self.dataset_input_sample
        pred = self.sync_send_prediction_request(network_input=net_input)
        self.NN_MO.position.value = self.NN_MO.rest_position.value + L * pred.reshape(self.NN_MO.position.shape)

    def onAnimateEndEvent(self, event):
        """
        Called within the Sofa pipeline at the end of the time step.
        :param event: Sofa Event
        :return: None
        """

        # Compute residual loss
        residual_loss = self.F_normalization_coef * np.sqrt(self.root.beamNN.ODESolver.squared_initial_residual)

        # Send training data
        if self.compute_essential_data:
            # Data are computed from FEM beam if not data from dataset
            net_input = self.F_normalization_coef * self.initial_force if self.compute_fem_solution else self.dataset_input_sample
            net_output = inv_L * (self.MO.position.value - self.MO.rest_position.value) if self.compute_fem_solution else self.dataset_output_sample
            self.sync_send_training_data(network_input=net_input,
                                         network_output=net_output)
            # Update mesh in vedo (if fem beam is not computed it stays to rest shape)
            if self.compute_fem_solution:
                positions = np.array(self.MO.position.value, dtype=float)
                self.sync_send_labeled_data(positions, 'positions')
                self.sync_send_labeled_data(residual_loss, 'loss')
        self.sync_send_command_done()  # Very important

        # Update step counter to use the next modal amplitude
        self.nb_step = (self.nb_step + 1) % len(modal_amplitude)

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
        return self.root.beamFEM.ODESolver.converged.value or not self.compute_fem_solution

    def applyPrediction(self, prediction):
        """
        Apply the prediction of the network in the Sofa environment. Automatically called by EnvironmentManager.
        :return: None
        """
        pass

    def setDatasetSample(self, sample_in, sample_out):
        """
        Use data from dataset
        :param sample_in:
        :param sample_out:
        :return:
        """
        self.dataset_input_sample = sample_in.reshape(self.input_size)
        self.dataset_output_sample = sample_out.reshape(self.output_size)
        # From now, fem solution is not computed anymore
        if self.compute_fem_solution:
            self.compute_fem_solution = False

    def close(self):
        print(f"Closing Env n°{self.instance_id}")

    def __str__(self):
        return f"Environment n°{self.instance_id} with tensor {self.tensor}"
