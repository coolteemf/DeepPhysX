from DeepPhysX_Core.Environment.BaseEnvironment import BaseEnvironment
from DeepPhysX_Core.Visualizer.VedoObjectFactories.VedoObjectFactory import VedoObjectFactory
from numpy import mean, pi, array
from numpy.random import random

import time


# This class generate a random vector between 0 and pi of 50 value and compute its mean.
# The vector is the input of the network and the groundtruth is the mean.
# The data are generated on the animate function
# The data are sent to the artificial neural network in the onStep function
# We added a visualizer hence we have to update the data at each step

class MeanEnvironment(BaseEnvironment):

    def __init__(self,
                 ip_address='localhost',
                 port=10000,
                 data_converter=None,
                 instance_id=1,
                 number_of_instances=1,
                 visualizer_class=None):
        BaseEnvironment.__init__(self,
                                 ip_address=ip_address,
                                 port=port,
                                 data_converter=data_converter,
                                 instance_id=instance_id,
                                 number_of_instances=number_of_instances,
                                 visualizer_class=visualizer_class)
        self.factory = VedoObjectFactory()

    def send_visualization(self):
        pos = pi * random((25, 2))
        # Point cloud
        self.factory.addObject(object_type="Points", data_dict={"positions": pos, "c": "blue", "at": self.instance_id, "r": 8})
        # Ground truth value
        self.factory.addObject(object_type="Points", data_dict={"positions": mean(pos, axis=0)[None, :], "c": "red", "at": self.instance_id, "r": 12})
        # Prediction value
        self.factory.addObject(object_type="Points", data_dict={"positions": mean(pos, axis=0)[None, :], "c": "green", "at": self.instance_id, "r": 12})
        return self.factory.objects_dict

    def send_parameters(self):
        return

    def create(self):
        print(f"Created client n°{self.instance_id}")

    async def animate(self):
        self.input = pi * random((25, 2))
        self.output = mean(self.input, axis=0)

    async def step(self):  # This function is called by a request of the server
        await self.animate()
        # get_prediction ask the neural network its output for the given input
        neural_network_prediction = await self.get_prediction(input_array=self.input)

        # Specify how to use the prediction
        self.apply_prediction(neural_network_prediction)
        await self.onStep()

    def apply_prediction(self, prediction):
        # In our case the prediction is only used to update the visual data
        # Point cloud
        self.factory.updateObject_dict(object_id=0, new_data_dict={'positions': self.input})
        # Ground truth value
        self.factory.updateObject_dict(object_id=1, new_data_dict={'position': self.output})
        # Prediction value
        self.factory.updateObject_dict(object_id=2, new_data_dict={'position': prediction})

    async def onStep(self):
        # Send visualisation data to update
        await self.update_visualisation(visu_dict=self.factory.updated_object_dict)
        # Send the training data
        await self.send_training_data(network_input=self.input, network_output=self.output)
