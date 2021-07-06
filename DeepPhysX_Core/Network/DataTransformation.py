class DataTransformation:

    def __init__(self, network_config):
        self.network_config = network_config

    def transformBeforePrediction(self, input, ground_truth):
        return input, ground_truth

    def transformAfterPrediction(self, output, ground_truth):
        return output, ground_truth
