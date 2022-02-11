from typing import Any, Optional, Dict

from numpy import array, ndarray

from DeepPhysX_Core.AsyncSocket.TcpIpClient import TcpIpClient


class BaseEnvironment(TcpIpClient):

    def __init__(self,
                 ip_address: str = 'localhost',
                 port: int = 10000,
                 instance_id: int = 1,
                 number_of_instances: int = 1,
                 as_tcp_ip_client: bool = True,
                 environment_manager: Any = None):
        """
        BaseEnvironment is an environment class to compute simulated data for the network and its optimization process.

        :param str ip_address: IP address of the TcpIpObject
        :param int port: Port number of the TcpIpObject
        :param int instance_id: ID of the instance
        :param int number_of_instances: Number of simultaneously launched instances
        :param bool as_tcp_ip_client: Environment is owned by a TcpIpClient if True, by an EnvironmentManager if False
        :param environment_manager: EnvironmentManager that handles the Environment if 'as_tcpip_client' is False
        """

        TcpIpClient.__init__(self,
                             instance_id=instance_id,
                             number_of_instances=number_of_instances,
                             as_tcp_ip_client=as_tcp_ip_client,
                             ip_address=ip_address,
                             port=port)

        # Input and output to give to the network
        self.input: ndarray = array([])
        self.output: ndarray = array([])
        # Variables to store samples from Dataset
        self.sample_in: Optional[ndarray] = None
        self.sample_out: Optional[ndarray] = None
        # Loss data
        self.loss_data: Any = None
        # Manager if the Environment is not a TcpIpClient
        self.environment_manager: Any = environment_manager

    ##########################################################################################
    ##########################################################################################
    #                                 Initializing Environment                               #
    ##########################################################################################
    ##########################################################################################

    def recv_parameters(self, param_dict: Dict[Any, Any]) -> None:
        """
        Exploit received parameters before scene creation.
        Not mandatory.

        :param dict param_dict: Dictionary of parameters
        :return:
        """
        pass

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

    def send_parameters(self) -> None:
        """
        Create a dictionary of parameters to send to the manager.
        Not mandatory.

        :return:
        """
        return

    def send_visualization(self) -> None:
        """
        Define the visualization objects to send to the Visualizer.
        Not mandatory.

        :return:
        """
        return

    ##########################################################################################
    ##########################################################################################
    #                                 Environment behavior                                   #
    ##########################################################################################
    ##########################################################################################

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
        return True

    def apply_prediction(self, prediction: ndarray) -> None:
        """
        Apply network prediction in environment.
        Not mandatory.

        :param ndarray prediction: Prediction data
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

    ##########################################################################################
    ##########################################################################################
    #                                   Defining a sample                                    #
    ##########################################################################################
    ##########################################################################################

    def set_training_data(self, input_array: ndarray, output_array: ndarray) -> None:
        """
        Set the training data to send to the TcpIpServer or the EnvironmentManager.

        :param ndarray input_array: Network input
        :param ndarray output_array: Network expected output
        :return:
        """

        # Training data is set if the Environment can compute data
        if self.compute_essential_data:
            self.input = input_array
            self.output = output_array

    def set_loss_data(self, loss_data: Any) -> None:
        """
        Set the loss data to send to the TcpIpServer or the EnvironmentManager.

        :param loss_data: Optional data to compute loss.
        :return:
        """

        # Training data is set if the Environment can compute data
        if self.compute_essential_data:
            self.loss_data = loss_data if type(loss_data) in [list, ndarray] else array([loss_data])

    def set_additional_in_dataset(self, label: str, data: Any) -> None:
        """
        Set additional input data fields to store in the dataset.

        :param str label: Name of the data field
        :param data: Data to store
        :return:
        """

        # Training data is set if the Environment can compute data
        if self.compute_essential_data:
            self.additional_inputs[label] = data if type(data) in [list, ndarray] else array([data])

    def set_additional_out_dataset(self, label: str, data: Any) -> None:
        """
        Set additional output data fields to store in the dataset.

        :param str label: Name of the data field
        :param data: Data to store
        :return:
        """

        # Training data is set if the Environment can compute data
        if self.compute_essential_data:
            self.additional_outputs[label] = data if type(data) in [list, ndarray] else array([data])

    def reset_additional_datasets(self) -> None:
        """
        Reset the additional dataset dictionaries.

        :return:
        """

        self.additional_inputs = {}
        self.additional_outputs = {}

    ##########################################################################################
    ##########################################################################################
    #                                   Available requests                                   #
    ##########################################################################################
    ##########################################################################################

    def get_prediction(self, input_array: ndarray) -> ndarray:
        """
        Request a prediction from Network.

        :param ndarray input_array: Network input
        :return:
        """

        # If Environment is a TcpIpClient, send request to the Server
        if self.as_tcp_ip_client:
            return TcpIpClient.request_get_prediction(self, input_array=input_array)

        # Otherwise, check the hierarchy of managers
        if self.environment_manager.data_manager is None:
            raise ValueError("Cannot request prediction if DataManager does not exist")
        elif self.environment_manager.data_manager.manager is None:
            raise ValueError("Cannot request prediction if Manager does not exist")
        elif self.environment_manager.data_manager.manager.network_manager is None:
            raise ValueError("Cannot request prediction if NetworkManager does not exist")
        # Get a prediction
        return self.environment_manager.data_manager.manager.network_manager.compute_online_prediction(
            network_input=input_array[None, ])

    def update_visualisation(self, visu_dict: Dict[int, Dict[str, Any]]) -> None:
        """
        Triggers the Visualizer update.

        :param dict visu_dict: Updated visualization data.
        :return:
        """

        # If Environment is a TcpIpClient, request to the Server
        if self.as_tcp_ip_client:
            TcpIpClient.request_update_visualization(self, visu_dict=visu_dict)
        # Otherwise, request to the EnvironmentManager
        else:
            self.environment_manager.update_visualizer({self.instance_id: visu_dict})

    def __str__(self) -> str:
        """
        :return: String containing information about the BaseEnvironment object
        """
        description = "\n"
        description += f"  {self.name}\n"
        description += f"    Name: {self.name} n°{self.instance_id}\n"
        description += f"    Comments:\n"
        description += f"    Input size:\n"
        description += f"    Output size:\n"
        return description
