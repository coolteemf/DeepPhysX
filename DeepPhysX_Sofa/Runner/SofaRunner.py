import Sofa

from DeepPhysX_Core.Pipelines.BaseRunner import BaseRunner


class SofaRunner(Sofa.Core.Controller, BaseRunner):

    def __init__(self,
                 network_config,
                 dataset_config,
                 environment_config,
                 session_name='default',
                 session_dir=None,
                 nb_steps=0,
                 record_inputs=False,
                 record_outputs=False,
                 *args, **kwargs):
        """
        BaseRunner is a pipeline defining the running process of an artificial neural network.
        It provide a highly tunable learning process that can be used with any machine learning library.

        :param network_config: Specialisation containing the parameters of the network manager
        :param dataset_config: Specialisation containing the parameters of the dataset manager
        :param environment_config: Specialisation containing the parameters of the environment manager
        :param str session_name: Name of the newly created directory if session_dir is not defined
        :param str session_dir: Name of the directory in which to write all the necessary data
        :param int nb_steps: Number of simulation step to play
        :param bool record_inputs: Save or not the input in a numpy file
        :param bool record_outputs: Save or not the output in a numpy file
        """

        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        BaseRunner.__init__(self,
                            network_config=network_config,
                            dataset_config=dataset_config,
                            environment_config=environment_config,
                            session_name=session_name,
                            session_dir=session_dir,
                            nb_steps=nb_steps,
                            record_inputs=record_inputs,
                            record_outputs=record_outputs)
        self.run_begin()
        self.root = self.manager.data_manager.environment_manager.environment.root
        self.root.addObject(self)

    def onAnimateEndEvent(self, event):
        """
        Called within the Sofa pipeline at the end of the time step.

        :param event: Sofa Event
        :return: None
        """

        if self.running_condition():
            self.sample_begin()
            # prediction, loss = self.predict(animate=False)
            prediction = self.predict(animate=False)
            self.manager.data_manager.environment_manager.environment.apply_prediction(prediction)
            self.sample_end()
        else:
            self.run_end()
