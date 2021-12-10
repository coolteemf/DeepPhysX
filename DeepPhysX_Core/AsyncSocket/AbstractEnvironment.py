class AbstractEnvironment:

    def __init__(self,
                 instance_id=1,
                 number_of_instances=1,
                 as_tcpip_client=True):
        """
        AbstractEnvironment gathers the environment API for TcpIpClient.

        :param int instance_id: ID of the instance
        :param int number_of_instances: Number of simultaneously launched instances
        :param as_tcpip_client: Environment is a TcpIpObject if True, is owned by an EnvironmentManager if False
        """

        self.name = self.__class__.__name__ + f"n°{instance_id}"

        if instance_id > number_of_instances:
            raise ValueError(f"[{self.name}] Instance ID ({instance_id}) is bigger than max instances "
                             f"({number_of_instances})")
        self.instance_id = instance_id
        self.number_of_instances = number_of_instances
        self.as_tcpip_client = as_tcpip_client

        self.input, self.output = None, None
        self.input_size, self.output_size = None, None
        self.additional_inputs, self.additional_outputs = {}, {}
        self.compute_essential_data = True

    def create(self):
        raise NotImplementedError

    def init(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def recv_parameters(self, param_dict):
        raise NotImplementedError

    def send_visualization(self):
        raise NotImplementedError

    def send_parameters(self):
        raise NotImplementedError

    def applyPrediction(self, prediction):
        raise NotImplementedError

    def setDatasetSample(self, sample_in, sample_out, additional_in=None, additional_out=None):
        raise NotImplementedError
