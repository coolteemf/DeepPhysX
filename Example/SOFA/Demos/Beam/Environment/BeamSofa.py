"""
BeamSofa
Simulation of a Beam with FEM computed deformations.
The SOFA simulation contains two models of a Beam:
    * one to apply forces and compute deformations
    * one to apply the network predictions
"""

# Python related imports
import os
import sys
import numpy as np

# Sofa & Caribou related imports
import Sofa.Simulation
import SofaRuntime

# DeepPhysX related imports
from DeepPhysX_Sofa.Environment.SofaEnvironment import SofaEnvironment

# Working session related imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from parameters import p_grid, p_forces


class BeamSofa(SofaEnvironment):

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
        self.f_grid_topo = None
        self.f_grid_mo = None
        self.f_surface_topo = None
        self.f_visu = None

        # Network objects
        self.n_grid_topo = None
        self.n_grid_mo = None
        self.n_surface_topo = None
        self.n_visu = None

        # Forces
        self.idx_surface = None
        self.cff_box = None
        self.cff = None

    def create(self):
        """
        Create the Sofa scene graph. Automatically called by SofaEnvironmentConfig.
        """

        # Add required plugins
        SofaRuntime.PluginRepository.addFirstPath(os.environ['CARIBOU_INSTALL'])
        plugins = ['SofaComponentAll', 'SofaCaribou', 'SofaBaseTopology', 'SofaGeneralEngine',
                   'SofaEngine', 'SofaOpenglVisual', 'SofaBoundaryCondition', 'SofaBaseCollision']
        self.root.addObject('RequiredPlugin', pluginName=plugins)

        # Scene visual style
        self.root.addObject('VisualStyle', displayFlags="showVisualModels showWireframe")

        # Create FEM and / or NN models
        if self.create_model['fem']:
            self.createFEM()
        if self.create_model['nn']:
            self.createNN()

    def createFEM(self):
        """
        FEM model of Liver. Used to apply forces and compute deformations.
        """

        # Create child node
        self.root.addChild('fem')

        # ODE solver + Static solver
        self.solver = self.root.fem.addObject('StaticODESolver', name='ODESolver', newton_iterations=20,
                                              printLog=False, correction_tolerance_threshold=1e-8,
                                              residual_tolerance_threshold=1e-8)
        self.root.fem.addObject('ConjugateGradientSolver', name='StaticSolver', preconditioning_method='Diagonal',
                                maximum_number_of_iterations=2000, residual_tolerance_threshold=1e-8, printLog=False)

        # Grid topology of the model
        self.f_grid_topo = self.root.fem.addObject('RegularGridTopology', name='GridTopo', min=p_grid.min.tolist(),
                                                   max=p_grid.max.tolist(), nx=p_grid.res[0], ny=p_grid.res[1],
                                                   nz=p_grid.res[2])
        self.f_grid_mo = self.root.fem.addObject('MechanicalObject', name='GridMO', src='@GridTopo', showObject=False)
        self.root.fem.addObject('HexahedronSetTopologyContainer', name='HexaTopo', src='@GridTopo')
        self.root.fem.addObject('HexahedronSetGeometryAlgorithms', template='Vec3d')
        self.root.fem.addObject('HexahedronSetTopologyModifier')

        # Material
        self.root.fem.addObject('NeoHookeanMaterial', name='NH', young_modulus=2000, poisson_ratio=0.45)
        self.root.fem.addObject('HyperelasticForcefield', template="Hexahedron", material="@NH",
                                topology='@HexaTopo', printLog=False)

        # Surface
        self.f_surface_topo = self.root.fem.addObject('QuadSetTopologyContainer', name='QuadTopo', src='@HexaTopo')
        self.root.fem.addObject('QuadSetGeometryAlgorithms', template='Vec3d')
        self.root.fem.addObject('QuadSetTopologyModifier')
        self.root.fem.addObject('Hexa2QuadTopologicalMapping', input='@HexaTopo', output='@QuadTopo')

        # Fixed section
        self.root.fem.addObject('BoxROI', name='FixedBox', box=p_grid.fixed_box.tolist(), drawBoxes=True, drawSize=1.)
        self.root.fem.addObject('FixedConstraint', indices='@FixedBox.indices')

        # Forces
        self.cff_box = self.root.fem.addObject('BoxROI', name='ForceBox', box=p_grid.size, drawBoxes=False,
                                               drawSize=1)
        self.cff = self.root.fem.addObject('ConstantForceField', name='CFF', showArrowSize=0.1,
                                           force=[0., 0., 0.], indices=list(iter(range(p_grid.nb_nodes))))

        # Visual
        self.root.fem.addChild('visual')
        self.f_visu = self.root.fem.visual.addObject('OglModel', src='@../GridTopo', color='yellow')
        self.root.fem.visual.addObject('BarycentricMapping', input='@../GridMO', output='@./')

    def createNN(self):
        """
        Network model of Liver. Used to apply predictions.
        """

        # Create child node
        self.root.addChild('nn')

        # Grid topology of the model
        self.n_grid_topo = self.root.nn.addObject('RegularGridTopology', name='GridTopo', min=p_grid.min.tolist(),
                                                  max=p_grid.max.tolist(), nx=p_grid.res[0], ny=p_grid.res[1],
                                                  nz=p_grid.res[2])
        self.n_grid_mo = self.root.nn.addObject('MechanicalObject', name='GridMO', src='@GridTopo', showObject=False)
        self.root.nn.addObject('HexahedronSetTopologyContainer', name='HexaTopo', src='@GridTopo')
        self.root.nn.addObject('HexahedronSetGeometryAlgorithms', template='Vec3d')
        self.root.nn.addObject('HexahedronSetTopologyModifier')

        # Surface
        self.n_surface_topo = self.root.nn.addObject('QuadSetTopologyContainer', name='QuadTopo', src='@HexaTopo')
        self.root.nn.addObject('QuadSetGeometryAlgorithms', template='Vec3d')
        self.root.nn.addObject('QuadSetTopologyModifier')
        self.root.nn.addObject('Hexa2QuadTopologicalMapping', input='@HexaTopo', output='@QuadTopo')

        # Fixed section
        if not self.create_model['fem']:
            self.root.nn.addObject('BoxROI', name='FixedBox', box=p_grid.fixed_box.tolist(), drawBoxes=False,
                                   drawSize=1.)
            self.root.nn.addObject('FixedConstraint', indices='@FixedBox.indices')

        # Forces
        if not self.create_model['fem']:
            self.cff_box = self.root.nn.addObject('BoxROI', name='ForceBox', box=p_grid.size, drawBoxes=True,
                                                  drawSize=1)
            self.cff = self.root.nn.addObject('ConstantForceField', name='CFF', showArrowSize=0.1,
                                              force=[0., 0., 0.], indices=list(iter(range(p_grid.nb_nodes))))

        # Visual
        self.root.nn.addChild('visual')
        self.n_visu = self.root.nn.visual.addObject('OglModel', src='@../GridTopo', color='red')
        self.root.nn.visual.addObject('BarycentricMapping', input='@../GridMO', output='@./')

    def onSimulationInitDoneEvent(self, event):
        """
        Called within the Sofa pipeline at the end of the scene graph initialisation.
        """

        # Get the indices of node on the surface
        surface = self.f_surface_topo if self.create_model['fem'] else self.n_surface_topo
        self.idx_surface = surface.quads.value.reshape(-1)

    def onAnimateBeginEvent(self, event):
        """
        Called within the Sofa pipeline at the beginning of the time step. Define force vector.
        """

        # Reset positions
        if self.create_model['fem']:
            self.f_grid_mo.position.value = self.f_grid_mo.rest_position.value
        if self.create_model['nn']:
            self.n_grid_mo.position.value = self.n_grid_mo.rest_position.value

        # Create a random box ROI, select nodes of the surface
        force_node = self.root.fem if self.create_model['fem'] else self.root.nn
        indices = []
        # Avoid empty box
        while len(indices) == 0:
            # Define random box
            x_min = np.random.randint(p_grid.min[0], p_grid.max[0] - 10)
            x_max = np.random.randint(x_min + 10, p_grid.max[0])
            y_min = np.random.randint(p_grid.min[1], p_grid.max[1] - 10)
            y_max = np.random.randint(y_min + 10, p_grid.max[1])
            z_min = np.random.randint(p_grid.min[2], p_grid.max[2] - 10)
            z_max = np.random.randint(z_min + 10, p_grid.max[2])
            # Set the new bounding box
            force_node.removeObject(self.cff_box)
            self.cff_box = force_node.addObject('BoxROI', name='ForceBox', drawBoxes=False, drawSize=1,
                                                box=[x_min, y_min, z_min, x_max, y_max, z_max])
            self.cff_box.init()
            # Get the intersection with the surface
            indices = list(self.cff_box.indices.value)
            indices = list(set(indices).intersection(set(self.idx_surface)))

        # Create a random force vector
        F = np.random.uniform(low=-1, high=1, size=(3,))
        K = np.random.randint(10, 20)
        F = K * (F / np.linalg.norm(F))

        # Update force field
        force_node.removeObject(self.cff)
        self.cff = force_node.addObject('ConstantForceField', name='CFF', showArrowSize=0.1, indices=indices,
                                        force=list(F))
        self.cff.init()

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
