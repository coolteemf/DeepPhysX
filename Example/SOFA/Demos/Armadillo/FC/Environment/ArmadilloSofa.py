"""
ArmadilloSofa
Simulation of an Armadillo with FEM computed deformations.
The SOFA simulation contains two models of an Armadillo :
    * one to apply forces and compute deformations
    * one to apply the network predictions
"""

# Python related imports
import os
import sys

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np

# Sofa related imports
import Sofa.Simulation
import SofaRuntime

# DeepPhysX related imports
from DeepPhysX_Sofa.Environment.SofaEnvironment import SofaEnvironment

# Session related imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from parameters import p_model, p_grid, p_forces


class ArmadilloSofa(SofaEnvironment):

    def __init__(self,
                 root_node,
                 ip_address='localhost',
                 port=10000,
                 instance_id=0,
                 number_of_instances=1,
                 as_tcp_ip_client=True,
                 environment_manager=None):

        SofaEnvironment.__init__(self,
                                 root_node=root_node,
                                 ip_address=ip_address,
                                 port=port,
                                 instance_id=instance_id,
                                 number_of_instances=number_of_instances,
                                 as_tcp_ip_client=as_tcp_ip_client,
                                 environment_manager=environment_manager)

        # With flag set to True, the model is created
        self.create_model = {'fem': True, 'nn': False}

        # FEM objects
        self.solver = None
        self.f_sparse_grid_mo = None
        self.f_force_grid_mo = None
        self.f_surface_mo = None
        self.f_visu = None

        # Network objects
        self.n_sparse_grid_mo = None
        self.n_force_grid_mo = None
        self.n_surface_mo = None
        self.n_visu = None

        # Forces
        self.cff = []
        self.sphere = []
        self.surface_node = None

    def create(self):
        """
        Create the Sofa scene graph. Automatically called by SofaEnvironmentConfig.
        """

        # Add required plugins
        SofaRuntime.PluginRepository.addFirstPath(os.environ['CARIBOU_INSTALL'])
        plugins = ['SofaComponentAll', 'SofaLoader', 'SofaCaribou', 'SofaBaseTopology', 'SofaGeneralEngine',
                   'SofaEngine', 'SofaOpenglVisual', 'SofaBoundaryCondition']
        self.root.addObject('RequiredPlugin', pluginName=plugins)

        # Scene visual style
        self.root.addObject('VisualStyle', displayFlags="showVisualModels showWireframe")

        # Armadillo OBJ
        self.root.addObject('MeshObjLoader', name='Mesh', filename=p_model.mesh, scale3d=p_model.scale3d)
        self.root.addObject('MeshObjLoader', name='MeshCoarse', filename=p_model.mesh_coarse,
                            scale3d=p_model.scale3d)

        # Create FEM and / or NN models
        if self.create_model['fem']:
            self.createFEM()
        if self.create_model['nn']:
            self.createNN()

    def createFEM(self):
        """
        FEM model of Armadillo. Used to apply forces and compute deformations.
        """

        # Create FEM node
        self.root.addChild('fem')

        # ODE solver + Static solver
        self.solver = self.root.fem.addObject('StaticODESolver', name='ODESolver', newton_iterations=20,
                                              printLog=False, correction_tolerance_threshold=1e-8,
                                              residual_tolerance_threshold=1e-8)
        self.root.fem.addObject('ConjugateGradientSolver', name='StaticSolver', preconditioning_method='Diagonal',
                                maximum_number_of_iterations=2000, residual_tolerance_threshold=1e-8, printLog=False)

        # Grid topology of the model
        self.root.fem.addObject('SparseGridTopology', name='SparseGridTopo', src='@../MeshCoarse', n=p_grid.resolution)
        self.f_sparse_grid_mo = self.root.fem.addObject('MechanicalObject', name='SparseGridMO', src='@SparseGridTopo')

        # Material
        self.root.fem.addObject('SaintVenantKirchhoffMaterial', name='StVK', young_modulus=1000, poisson_ratio=0.45)
        self.root.fem.addObject('HyperelasticForcefield', template="Hexahedron", material="@StVK",
                                topology='@SparseGridTopo', printLog=False)

        # Fixed section
        self.root.fem.addObject('BoxROI', name='FixedBox', box=p_model.fixed_box, drawBoxes=True)
        self.root.fem.addObject('FixedConstraint', indices='@FixedBox.indices')

        # Force grid
        self.root.fem.addChild('forces')
        self.root.fem.forces.addObject('SparseGridTopology', name='ForceGridTopo', src='@../../MeshCoarse',
                                       n=p_grid.resolution)
        self.f_force_grid_mo = self.root.fem.forces.addObject('MechanicalObject', name='ForceGridMO',
                                                              src='@ForceGridTopo')
        self.root.fem.forces.addObject('BarycentricMapping', input='@../SparseGridMO', output='@./')

        # Surface
        self.root.fem.forces.addChild('surface')
        self.root.fem.forces.surface.addObject('TriangleSetTopologyContainer', name='SurfaceTopo',
                                               src='@../../../MeshCoarse')
        self.f_surface_mo = self.root.fem.forces.surface.addObject('MechanicalObject', name='SurfaceMO',
                                                                   src='@SurfaceTopo')
        self.root.fem.forces.surface.addObject('BarycentricMapping', input='@../ForceGridMO', output='@./')

        # Forces
        self.create_forces(self.root.fem.forces.surface)

        # Visual
        self.root.fem.addChild('visual')
        self.f_visu = self.root.fem.visual.addObject('OglModel', src='@../../Mesh', color='green')
        self.root.fem.visual.addObject('BarycentricMapping', input='@../SparseGridMO', output='@./')

    def createNN(self):
        """
        Network model of Armadillo. Used to apply predictions.
        """

        # Create Neural Network node
        self.root.addChild('nn')

        # Fake solvers (required to compute forces on the grid)
        self.root.nn.addObject('StaticODESolver', name='ODESolver', newton_iterations=0)
        self.root.nn.addObject('ConjugateGradientSolver', name='StaticSolver', maximum_number_of_iterations=0)

        # Grid topology
        self.root.nn.addObject('SparseGridTopology', name='SparseGridTopo', src='@../MeshCoarse', n=p_grid.resolution)
        self.n_sparse_grid_mo = self.root.nn.addObject('MechanicalObject', name='SparseGridMO', src='@SparseGridTopo')

        # Fixed section
        if not self.create_model['fem']:
            self.root.nn.addObject('BoxROI', name='FixedBox', box=p_model.fixed_box, drawBoxes=True)
            self.root.nn.addObject('FixedConstraint', indices='@FixedBox.indices')

        # Surface
        self.root.nn.addChild('surface')
        self.root.nn.surface.addObject('TriangleSetTopologyContainer', name='SurfaceTopo', src='@../../../MeshCoarse')
        self.n_surface_mo = self.root.nn.surface.addObject('MechanicalObject', name='SurfaceMO', src='@SurfaceTopo')
        self.root.nn.surface.addObject('BarycentricMapping', input='@../SparseGridMO', output='@./')

        # Forces
        if not self.create_model['fem']:
            self.create_forces(self.root.nn.surface)

        # Visual
        self.root.nn.addChild('visual')
        self.n_visu = self.root.nn.visual.addObject('OglModel', src='@../../Mesh', color='red')
        self.root.nn.visual.addObject('BarycentricMapping', input='@../SparseGridMO', output='@./')

    def create_forces(self, node):
        """
        Generate the force fields on specific areas.
        """

        # ConstantForceFields are applied on the surface of the object
        self.surface_node = node
        for zone in p_forces.zones:
            self.sphere.append(node.addObject('SphereROI', name=f'Sphere_{zone}', radii=p_forces.radius[zone],
                                              centers=p_forces.centers[zone], drawPoints=True, drawSize=1))
            self.cff.append(node.addObject('ConstantForceField', name=f'cff_{zone}', force=[0., 0., 0.],
                                           indices=f'@Sphere_{zone}.indices'))

    def onSimulationInitDoneEvent(self, event):
        """
        Called within the Sofa pipeline at the end of the scene graph initialisation.
        """

        # Delete ROI spheres
        for sphere in self.sphere:
            self.surface_node.removeObject(sphere)

    def onAnimateBeginEvent(self, event):
        """
        Called within the Sofa pipeline at the beginning of the time step. Define force vector.
        """

        # Reset positions
        if self.create_model['fem']:
            self.f_sparse_grid_mo.position.value = self.f_sparse_grid_mo.rest_position.value
        if self.create_model['nn']:
            self.n_sparse_grid_mo.position.value = self.n_sparse_grid_mo.rest_position.value

        # Reset forces
        for cff in self.cff:
            cff.force.value = np.array([0., 0., 0.])

        # Generate new forces
        zones = np.random.choice(len(self.cff), size=p_forces.simultaneous, replace=False)
        for i in zones:
            f = np.random.uniform(low=-1, high=1, size=(3,))
            f = f * p_forces.amplitude[p_forces.zones[i]]
            self.cff[i].force.value = f
            self.cff[i].showArrowSize.value = 10 * len(self.cff[i].forces.value)

    def onAnimateEndEvent(self, event):
        """
        Called within the Sofa pipeline at the end of the time step.
        """

        # Check whether if the solver diverged or not
        if not self.check_sample():
            print("Solver diverged.")

    def check_sample(self):
        """
        Check if the produced sample is correct. Automatically called by DeepPhysX to check sample validity.
        """

        # Check if the solver converged while computing FEM
        if self.create_model['fem']:
            if not self.solver.converged.value:
                # Reset simulation if solver diverged to avoid unwanted behaviour in following samples
                Sofa.Simulation.reset(self.root)
            return self.solver.converged.value
        return True

    def close(self):
        """
        Shutdown procedure.
        """

        print("Bye!")
