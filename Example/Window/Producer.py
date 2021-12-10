from DeepPhysX_Core.Environment.BaseEnvironment import BaseEnvironment
from DeepPhysX_Core.Visualizer.VedoObjectFactories.VedoObjectFactory import VedoObjectFactory
from numpy import mean, pi, random
import numpy as np
import time


# This class generate a random vector 2D vector and compute its polar representation
# The vector is the input of the network and the groundtruth is its polar representation.
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
        self.input = random.default_rng().uniform(low=-1, high=1, size=(2,)) / np.sqrt(2)
        self.output = self.input
        # Point cloud
        self.factory.addObject(object_type="Points", data_dict={"positions": self.input, "c": "blue", "at": self.instance_id, "r": 8})
        # Ground truth value
        self.factory.addObject(object_type="Points", data_dict={"positions": self.output, "c": "green", "at": self.instance_id, "r": 12})
        # Prediction truth value
        self.factory.addObject(object_type="Points", data_dict={"positions": self.output, "c": "red", "at": self.instance_id, "r": 12})
        # Window to display in polar coordinate value
        self.factory.addObject(object_type="Window", data_dict={"objects_id": [0, 1, 2], "axes": 12})
        return self.factory.objects_dict

    def send_parameters(self):
        return

    def create(self):
        print(f"Created client n°{self.instance_id}")

    async def animate(self):
        self.input = random.default_rng().uniform(low=-1, high=1, size=(2,))
        self.output = np.array([np.linalg.norm(self.input), np.arctan2(self.input[1], self.input[0])])

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
        # Cast polar to Cartesian :->
        self.factory.updateObject_dict(object_id=1, new_data_dict={'position': np.array([self.output[0]*np.cos(self.output[1]), self.output[0]*np.sin(self.output[1])])})
        # Prediction value
        self.factory.updateObject_dict(object_id=2, new_data_dict={'position':  np.array([prediction[0]*np.cos(prediction[1]), prediction[0]*np.sin(prediction[1])])})

    async def onStep(self):
        # Send visualisation data to update
        await self.update_visualisation(visu_dict=self.factory.updated_object_dict)
        # Send the training data
        await self.send_training_data(network_input=self.input, network_output=self.output)
