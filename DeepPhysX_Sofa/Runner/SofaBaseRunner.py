import Sofa


class SofaRunner(Sofa.Core.Controller):

    def __init__(self, runner, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.runner = runner
        self.runner.runBegin()

    def execute(self):
        self.runner.sampleBegin()
        prediction, loss = self.runner.predict(animate=False)
        self.runner.manager.environment_manager.environment.applyPrediction(prediction)
        self.runner.sampleEnd()

    def onAnimateBeginEvent(self, event):
        if self.runner.runningCondition():
            self.execute()
        else:
            self.runner.runEnd()
