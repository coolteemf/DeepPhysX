import copy

from Sandbox.Beam.BeamConfig.MouseForceManager import MouseForceManager
from Sandbox.Beam.NNBeam.NNBeam import NNBeam


class NNBeamDynamic(NNBeam):

    def __init__(self, root_node, config, idx_instance=1, visualizer_class=None):
        super(NNBeamDynamic, self).__init__(root_node, config, idx_instance, visualizer_class)
        self.config = config
        self.last_U = None

    def createBehavior(self, config):
        self.root.addObject('VisualStyle', displayFlags="showCollisionModels hideVisualModels")

        self.MO = self.root.beamNN.addObject('MechanicalObject', src='@Grid', name='MO', template='Vec3d',
                                             showObject=True)

        self.root.addChild('sphere')
        self.root.sphere.addObject('MechanicalObject', name="MO", position=[100., 25., 25.])
        self.root.sphere.addObject('UniformMass', totalMass=10)
        self.root.sphere.addObject('EulerImplicitSolver', name='ODESolver')
        self.root.sphere.addObject('CGLinearSolver', name='LinearSolver', iterations=100, tolerance=1.0e-8,
                                   threshold=1.0e-8)
        self.root.sphere.addObject('SphereModel', radius=1)
        self.root.sphere.addObject('StiffSpringForceField', template='Vec3d', printLog=True,
                                   stiffness=300, damping=1, length=0.,
                                   indices1=[0], object1='@sphere/MO',
                                   indices2=[3999], object2='@beamNN/MO')

        self.root.addChild("floor")
        self.root.floor.addObject('MeshObjLoader', name="loader", filename='mesh/floorFlatTri.obj')
        # self.root.floor.addObject('MeshObjLoader', name="loader", filename='mesh/floor3.obj')
        self.root.floor.addObject('MeshTopology', src='@loader')
        self.root.floor.addObject('MechanicalObject', src='@loader', template='Vec3d', showObject=False)
        self.root.floor.addObject('PointCollisionModel', moving=False, simulated=False, )
        self.root.floor.addObject('TriangleCollisionModel', bothSide=True, moving=False, simulated=False,
                                  contactStiffness=100)
        self.root.floor.addObject('LineCollisionModel', moving=False, simulated=False)

    def onSimulationInitDoneEvent(self, event):
        NNBeam.onSimulationInitDoneEvent(self, event)
        self.mouseManager = MouseForceManager(self.grid, [2, 2, 2], self.surface)

    def onAnimateBeginEvent(self, event):
        # self.MO.position.value = self.MO.rest_position.value
        pass

    def computeInput(self):
        F = copy.copy(self.MO.force.value)
        node = self.mouseManager.find_picked_node(F)
        if node is not None:
            F[node] = self.mouseManager.scale_max_force(F[node])
            F, local = self.mouseManager.distribute_force(node, F, 2.5, 8)
        self.input = F * self.root.dt.value ** 2
        # print(np.linalg.norm(F))

    def applyPrediction(self, prediction):
        u0 = prediction[0] * 10
        self.MO.position.value = u0 + self.MO.rest_position.array()

