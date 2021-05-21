from DeepPhysX_Sofa.Environment.SofaBaseEnvironment import SofaBaseEnvironment

import SofaRuntime

class MyEnvironment(SofaBaseEnvironment):

    def __init__(self, root_node, simulations_per_step=1, max_wrong_samples=10, idx_instance=1, *args, **kwargs):
        SofaBaseEnvironment.__init__(self, root_node, simulations_per_step, max_wrong_samples, idx_instance, *args, **kwargs)
        self.val = idx_instance
        self.inputSize, self.outputSize = 3, 3
        self.descriptionName = "MyEnvironment"

    def create(self):
        SofaRuntime.importPlugin('SofaBaseMechanics')
        self.root.gravity = [0, -9.81, 0]
        self.root.dt = 0.01
        self.root.addObject('DefaultVisualManagerLoop')
        self.root.addObject('DefaultAnimationLoop')
        # Add new nodes and objects in the scene
        self.node = self.root.addChild("Node1")
        self.MO = self.node.addObject("MechanicalObject", name="DOF", template="Rigid3d", position="0 0 0   0 0 0 1",
                                      showObject="1")
        print("Scene created.")

    def onAnimateBeginEvent(self, event):
        with self.MO.position.writeable() as MO:
            MO[0][0:3] = MO[0][0:3] + [self.val, self.val, self.val]
        self.inputs = self.MO.position.value[0][0:3]

    def onReset(self):
        with self.MO.position.writeable() as MO:
            MO[0][0:3] = [0, 0, 0]

    def getDescription(self):
        SofaBaseEnvironment.getDescription(self)
        self.description += "   Index: {}\n".format(self.val)
        return self.description
