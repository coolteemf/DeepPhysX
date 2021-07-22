from DeepPhysX_Core.Pipelines.BasePipeline import BasePipeline
from DeepPhysX_Core.Manager.Manager import Manager
from DeepPhysX_Core.Network.BaseNetworkConfig import BaseNetworkConfig
from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX_Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig


class BaseRunner(BasePipeline):

    def __init__(self, network_config: BaseNetworkConfig, dataset_config: BaseDatasetConfig,
                 environment_config: BaseEnvironmentConfig, session_name='default', session_dir=None,
                 nb_steps=0, record_inputs=False, record_outputs=False):
        """
        BaseRunner is a pipeline defining the running process of an artificial neural network.
        It provide a highly tunable learning process that can be used with any machine learning library.

        :param BaseNetworkConfig network_config: Specialisation containing the parameters of the network manager
        :param BaseDatasetConfig dataset_config: Specialisation containing the parameters of the dataset manager
        :param BaseEnvironmentConfig environment_config: Specialisation containing the parameters of the environment manager
        :param str session_name: Name of the newly created directory if session_dir is not defined
        :param str session_dir: Name of the directory in which to write all of the neccesary data
        :param int nb_steps: Number of simulation step to play
        :param bool record_inputs: Save or not the input in a numpy file
        :param bool record_outputs: Save or not the output in a numpy file
        """

        BasePipeline.__init__(self, pipeline='prediction')

        self.name = self.__class__.__name__

        # Check the arguments
        if not isinstance(network_config, BaseNetworkConfig):
            raise TypeError("[BaseRunner] The network configuration must be a BaseNetworkConfig")
        if not isinstance(environment_config, BaseEnvironmentConfig):
            raise TypeError("[BaseRunner] The environment configuration must be a BaseEnvironmentConfig")
        if dataset_config is not None and not isinstance(dataset_config, BaseDatasetConfig):
            raise TypeError("[BaseRunner] The dataset configuration must be a BaseDatasetConfig")
        if type(session_name) != str:
            raise TypeError("[BaseRunner] The network config must be a BaseNetworkConfig object.")
        if session_dir is not None and type(session_dir) != str:
            raise TypeError("[BaseRunner] The session directory must be a str.")
        if type(nb_steps) != int or nb_steps < 0:
            raise TypeError("[BaseRunner] The number of steps must be a positive int")

        self.nb_samples = nb_steps
        self.idx_step = 0

        # Tell if data is recording while predicting (output is recorded only if input too)
        self.record_data = (record_inputs, record_outputs and record_inputs)

        self.manager = Manager(pipeline=self, network_config=network_config, dataset_config=dataset_config,
                               environment_config=environment_config, session_name=session_name,
                               session_dir=session_dir)

    def execute(self):
        """
        Main function of the running process \"execute\" call the functions associated with the learning process.
        Each of the called functions are already implemented so one can start a basic run session.
        Each of the called function can also be rewritten via inheritance to provide more specific / complex running process.

        :return:
        """
        self.runBegin()
        while self.runningCondition():
            self.sampleBegin()
            prediction, loss_value = self.predict()
            self.manager.data_manager.environment_manager.environment.applyPrediction(prediction)
            self.sampleEnd()
        self.runEnd()

    def predict(self, animate=True):
        """
        Pull the data from the manager and return the prediction

        :param bool animate: True if getData fetch from the environment
        :return: tuple (numpy.ndarray, float)
        """
        self.manager.getData(animate=animate)
        return self.manager.getPrediction()

    def runBegin(self):
        """
        Called once at the very beginning of the Run process.
        Allows the user to run some pre-computations.

        :return:
        """
        pass

    def runEnd(self):
        """
        Called once at the very end of the Run process.
        Allows the user to run some post-computations.

        :return:
        """
        pass

    def runningCondition(self):
        """
        Condition that characterize the end of the runnning process

        :return: bool : False if the training needs to stop.
        """
        running = self.idx_step < self.nb_samples if self.nb_samples > 0 else True
        self.idx_step += 1
        return running

    def sampleBegin(self):
        """
        Called one at the start of each step.
        Allows the user to run some pre-step computations.

        :return:
        """
        pass

    def sampleEnd(self):
        """
        Called one at the end of each step.
        Allows the user to run some post-step computations.

        :return:
        """
        pass

    def close(self):
        """
        End the running process and close all the managers

        :return:
        """
        self.manager.close()

    def __str__(self):
        """

        :return: str Contains running informations about the running process
        """
        description = ""
        description += f"Running statistics :\n"
        description += f"Number of simulation step: {self.nb_samples}\n"
        description += f"Record inputs : {self.record_data[0]}\n"
        description += f"Record outputs : {self.record_data[1]}\n"

        return description