class AbstractEnvironment:

    def __init__(self, instance_id=1):
        """
        AbstractEnvironment gathers the environment API for TcpIpClient.

        :param int instance_id: ID of the instance
        """

        self.name = self.__class__.__name__ + f"n°{instance_id}"
        self.instance_id = instance_id
        self.input, self.output = None, None
        self.input_size, self.output_size = None, None

    def create(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def computeInput(self):
        raise NotImplementedError

    def computeOutput(self):
        raise NotImplementedError

    def checkSample(self, check_input=True, check_output=True):
        raise NotImplementedError

    def getInput(self):
        raise NotImplementedError

    def getOutput(self):
        raise NotImplementedError

    def recv_parameters(self, param_dict):
        raise NotImplementedError

    def send_parameters(self):
        raise NotImplementedError
