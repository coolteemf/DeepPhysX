class DataTransformation:

    def __init__(self, network_config):
        self.network_config = network_config
        self.data_type = any

    @staticmethod
    def check_type(func):
        def inner(self, *args):
            for data in args:
                if type(data) != self.data_type:
                    raise TypeError(f"[DataTransformation] The given data must be of type {self.data_type}, "
                                    f"found {type(data)}")
            return func(self, *args)
        return inner

    def transformBeforePrediction(self, data_in):
        return data_in

    def transformBeforeLoss(self, data_out, data_gt):
        return data_out, data_gt

    def transformBeforeApply(self, data_out):
        return data_out
