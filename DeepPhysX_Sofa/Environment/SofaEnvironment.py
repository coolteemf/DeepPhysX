from numpy import ndarray

from DeepPhysX_Core.Environment.BaseEnvironment import BaseEnvironment

import Sofa
import Sofa.Simulation


class SofaEnvironment(Sofa.Core.Controller, BaseEnvironment):

    def __init__(self,
                 root_node,
                 ip_address='localhost',
                 port=10000,
                 instance_id=0,
                 number_of_instances=1,
                 as_tcp_ip_client=True,
                 environment_manager=None,
                 *args, **kwargs):
        """
        SofaEnvironment is an environment class base on SOFA to compute simulated data for the network and its
        optimization process.

        :param root_node: Sofa.Core.Node used to create the scene graph
        :param str ip_address: IP address of the TcpIpObject
        :param int port: Port number of the TcpIpObject
        :param int instance_id: ID of the instance
        :param int number_of_instances: Number of simultaneously launched instances
        :param bool as_tcp_ip_client: Environment is owned by a TcpIpClient if True, by an EnvironmentManager if False
        :param environment_manager: EnvironmentManager that handles the Environment if 'as_tcpip_client' is False
        """

        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        # Warning: Define root node before init Environment
        self.root = root_node
        BaseEnvironment.__init__(self, ip_address=ip_address, port=port, instance_id=instance_id,
                                 number_of_instances=number_of_instances, as_tcp_ip_client=as_tcp_ip_client,
                                 environment_manager=environment_manager)

    def create(self):
        """
        Create the environment given the configuration.
        Must be implemented by user.

        :return:
        """
        raise NotImplementedError

    def init(self):
        """
        Initialize the root node.

        :return:
        """
        Sofa.Simulation.init(self.root)

    async def step(self):
        """
        Compute the number of steps specified by simulations_per_step

        :return:
        """
        await self.animate()
        await self.on_step()

    async def animate(self):
        """
        Trigger an Animation step.

        :return:
        """
        Sofa.Simulation.animate(self.root, self.root.dt.value)

    async def on_step(self):
        """
        Executed after an animation step.

        :return:
        """
        pass

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

    def init_visualizer(self):
        """
        Init visualization data.

        :return:
        """
        pass

    def __str__(self):
        """
        :return: String containing information about the BaseEnvironmentConfig object
        """
        description = BaseEnvironment.__str__(self)
        return description
