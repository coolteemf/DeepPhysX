class DataTransformation:

    def __init__(self, network_config):
        """
        DataTransformation is dedicated to data operations before and after network predictions.

        :param network_config: BaseNetworkConfig class
        """
        self.name = self.__class__.__name__

        self.network_config = network_config
        self.data_type = any

    @staticmethod
    def check_type(func):
        def inner(self, *args):
            for data in args:
                if type(data) != self.data_type:
                    raise TypeError(f"[{self.name}] Wrong data type: {self.data_type} required, get {type(data)}")
            return func(self, *args)
        return inner

    def transformBeforePrediction(self, data_in):
        """
        Apply data operations before network's prediction.

        :param data_in: Input data
        :return: Transformed input data
        """
        return data_in

    def transformBeforeLoss(self, data_out, data_gt):
        """
        Apply data operations between network's prediction and loss computation.

        :param data_out: Prediction data
        :param data_gt: Ground truth data
        :return: Transformed prediction data, transformed ground truth data
        """
        return data_out, data_gt

    def transformBeforeApply(self, data_out):
        """
        Apply data operations between loss computation and prediction apply in environment.

        :param data_out: Prediction data
        :return: Transformed prediction data
        """
        return data_out

    def __str__(self):
        description = "\n"
        description += f"  {self.name}\n"
        description += f"    Data type: {self.data_type}\n"
        description += f"    Transformation before prediction: Identity\n"
        description += f"    Transformation before loss: Identity\n"
        description += f"    Transformation before apply: Identity\n"
        return description

