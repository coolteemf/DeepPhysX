class AbstractEnvironment:

    def __init__(self, instance_id=1, number_of_instances=1, as_tcpip_client=True):
        """
        AbstractEnvironment gathers the environment API for TcpIpClient.

        :param int instance_id: ID of the instance
        """
        self.name = self.__class__.__name__ + f"n°{instance_id}"
        self.instance_id = instance_id

        if instance_id < number_of_instances:
            self.number_of_instances = number_of_instances
        else:
            raise ValueError(f"Instance ID ({instance_id}) is bigger than max instances ({number_of_instances})")

        self.as_tcpip_client = as_tcpip_client

        self.input, self.output = None, None
        self.input_size, self.output_size = None, None
        self.compute_essential_data = True

    def create(self):
        raise NotImplementedError

    def init(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def checkSample(self, check_input=True, check_output=True):
        raise NotImplementedError

    def recv_parameters(self, param_dict):
        raise NotImplementedError

    def send_visualization(self):
        raise NotImplementedError

    def send_parameters(self):
        raise NotImplementedError

    def applyPrediction(self, prediction):
        raise NotImplementedError

    def setDatasetSample(self, sample_in, sample_out):
        raise NotImplementedError
