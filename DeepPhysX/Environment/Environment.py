class Environment:

    def __init__(self, simulations_per_step=1):
        self.simulationPerStep = simulations_per_step
        self.inputs = None
        self.outputs = None
        self.inputSize = None
        self.outputSize = None

    def create(self):
        pass

    def reset(self):
        pass

    def step(self):
        raise NotImplementedError

    def getInput(self):
        return self.inputs

    def getOutput(self):
        return self.outputs
