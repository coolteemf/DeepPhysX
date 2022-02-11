from typing import Optional, Dict, Any
import numpy


class AbstractEnvironment:

    def __init__(self,
                 instance_id: int = 1,
                 number_of_instances: int = 1,
                 as_tcp_ip_client: bool = True):
        """
        AbstractEnvironment gathers the Environment API for TcpIpClient. Do not use AbstractEnvironment to implement
        your own Environment, use BaseEnvironment instead.

        :param int instance_id: ID of the instance
        :param int number_of_instances: Number of simultaneously launched instances
        :param bool as_tcp_ip_client: Environment is a TcpIpObject if True, is owned by an EnvironmentManager if False
        """

        self.name: str = self.__class__.__name__ + f" n°{instance_id}"

        if instance_id > number_of_instances:
            raise ValueError(f"[{self.name}] Instance ID ({instance_id}) is bigger than max instances "
                             f"({number_of_instances})")
        self.instance_id: int = instance_id
        self.number_of_instances: int = number_of_instances
        self.as_tcp_ip_client: bool = as_tcp_ip_client

        # Input and output to give to the network
        self.input: numpy.ndarray = numpy.array([])
        self.output: numpy.ndarray = numpy.array([])
        self.loss_data: Any = None
        # Variables to store samples from Dataset
        self.sample_in: Optional[numpy.ndarray] = None
        self.sample_out: Optional[numpy.ndarray] = None
        self.additional_inputs: Dict[str, Any] = {}
        self.additional_outputs: Dict[str, Any] = {}
        self.compute_essential_data: bool = True

    def create(self) -> None:
        """
        Create the Environment.
        Must be implemented by user.

        :return:
        """
        raise NotImplementedError

    def init(self) -> None:
        """
        Initialize the Environment.
        Not mandatory.

        :return:
        """
        pass

    def close(self) -> None:
        """
        Close the Environment.
        Not mandatory.

        :return:
        """
        pass

    async def step(self) -> None:
        """
        Compute the number of steps in the Environment specified by simulations_per_step in EnvironmentConfig.
        Must be implemented by user.

        :return:
        """
        raise NotImplementedError

    def check_sample(self, check_input: bool = True, check_output: bool = True) -> bool:
        """
        Check if the current produced sample is usable for training.
        Not mandatory.

        :param bool check_input: If True, input tensors need to be checked
        :param bool check_output: If True, output tensors need to be checked
        :return: bool - Current data can be used or not
        """
        raise NotImplementedError

    def recv_parameters(self, param_dict: Dict[Any, Any]) -> None:
        """
        Called before create and init, receive simulation parameters from the server

        :param param_dict: Dictionary of parameters
        """
        raise NotImplementedError

    def send_visualization(self) -> dict:
        """
        Define the visualization objects to send to the Visualizer.
        Not mandatory.

        :return:
        """
        raise NotImplementedError

    def send_parameters(self) -> dict:
        """
        Create a dictionary of parameters to send to the manager.
        Not mandatory.

        :return:
        """
        raise NotImplementedError

    def apply_prediction(self, prediction: numpy.ndarray) -> None:
        """
        Apply network prediction in environment.
        Not mandatory.

        :param ndarray prediction: Prediction data
        :return:
        """
        raise NotImplementedError
