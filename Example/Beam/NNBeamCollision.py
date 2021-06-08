import copy
from Example.Beam.MouseForceManager import MouseForceManager

from Example.Beam.NNBeam import NNBeam


class NNBeamCollision(NNBeam):

    def __init__(self, root_node, config, idx_instance=1, training=True):
        super(NNBeamCollision, self).__init__(root_node, config, idx_instance)
        self.config = config

    def createBehavior(self, p_grid):


        self.root.addObject('DefaultPipeline')
        self.root.addObject('FreeMotionAnimationLoop')
        self.root.addObject('GenericConstraintSolver', tolerance="1e-6", maxIterations="10")
        self.root.addObject('BruteForceDetection')
        # self.root.addObject('RuleBasedContactManager', responseParams="mu=" + str(0.0), name='Response',
        #                     response='FrictionContact')
        # self.root.addObject('LocalMinDistance', alarmDistance=10, contactDistance=5, angleCone=0.01)


        self.MO = self.root.BeamNN.addObject('MechanicalObject', src='@Grid', name='MO', template='Vec3d',
                                             showObject=True)
        # Ball
        self.root.addChild('BallFEM')
        self.root.BallFEM.addObject('EulerImplicitSolver', name='DynamicSolver')
        self.root.BallFEM.addObject('ConjugateGradientSolver', name='LinearSolver')
        self.root.BallFEM.addObject('MechanicalObject', name='MO', showObject=True, template='Rigid3d',
                                    position=[50, 50, 50, 0, 0, 0, 1])
        self.root.BallFEM.addObject('UniformMass', totalMass=10)
        # Collision
        self.root.BallFEM.addChild('Collision')
        self.root.BallFEM.Collision.addObject('MeshObjLoader', name="loader", filename="mesh/ball.obj",
                                              scale=5)
        self.root.BallFEM.Collision.addObject('MechanicalObject', src='@loader', template='Vec3d', showObject=False)
        self.root.BallFEM.Collision.addObject('PointCollisionModel')
        self.root.BallFEM.Collision.addObject('RigidMapping')

    def onSimulationInitDoneEvent(self, event):
        NNBeam.onSimulationInitDoneEvent(self, event)
        self.mouseManager = MouseForceManager(self.grid, [2, 2, 2], self.surface)

    def onAnimateBeginEvent(self, event):
        self.MO.position.value = self.MO.rest_position.value

    def computeInput(self):
        F = copy.copy(self.MO.force.value)
        node = self.mouseManager.find_picked_node(F)
        if node is not None:
            F[node] = self.mouseManager.scale_max_force(F[node])
            F, local = self.mouseManager.distribute_force(node, F, 2.5, 8)
        self.input = F

    def applyPrediction(self, prediction):
        u0 = prediction[0] * 10
        self.MO.position.value = u0 + self.MO.rest_position.array()
