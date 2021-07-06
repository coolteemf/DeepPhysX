class BasePipeline:

    def __init__(self, pipeline=None):
        self.type = pipeline    # Either training or prediction
        self.new_session = True
        self.record_data = None  # Can be of type {'in': bool, 'out': bool}
