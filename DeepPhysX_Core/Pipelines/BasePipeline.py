class BasePipeline:

    def __init__(self, pipeline=None):
        """
        Base class defining Pipelines common variables

        :param str pipeline: Values at either 'training' or 'prediction'
        """
        self.type = pipeline    # Either training or prediction
        self.new_session = True
        self.record_data = None  # Can be of type {'in': bool, 'out': bool}
