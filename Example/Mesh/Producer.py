from DeepPhysX_Core.Environment.BaseEnvironment import BaseEnvironment
from DeepPhysX_Core.Visualizer.VedoObjectFactories.VedoObjectFactory import VedoObjectFactory
import numpy as np
from vedo import Mesh
import os


# This class generate a random position in the unit cube ([-1, -1, -1], [1, 1, 1]) (INPUT)
# And compare its signed distance relative to the unit sphere. (OUTPUT)
# The data are generated on the animate function
# The data are sent to the artificial neural network in the onStep function

class MeanEnvironment(BaseEnvironment):

    def __init__(self,
                 ip_address='localhost',
                 port=10000,
                 instance_id=1,
                 number_of_instances=1,
                 visualizer_class=None):
        BaseEnvironment.__init__(self,
                                 ip_address=ip_address,
                                 port=port,
                                 instance_id=instance_id,
                                 number_of_instances=number_of_instances,
                                 visualizer_class=visualizer_class)
        self.factory = VedoObjectFactory()
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.mesh = Mesh(f"{dir_path}/ico.obj")

    def send_visualization(self):

        # Mesh
        self.factory.addObject(object_type="Mesh", data_dict={"positions": self.mesh.points().tolist(), "cells": self.mesh.cells(),  "c": "grey", "at": self.instance_id, "alpha": 0.5})

        # Target
        self.factory.addObject(object_type="Points", data_dict={"positions": [[1, 1, 0]], "c": "red", "at": self.instance_id, "r": 12})

        # We will input the distance in the Y value to move them both on the same axis, thus displaying the error
        # Real distance to the mesh
        self.factory.addObject(object_type="Points", data_dict={"positions": [[1, 0, 1]], "c": "blue", "at": self.instance_id, "r": 20})
        # Predicted distance to the mesh
        self.factory.addObject(object_type="Points", data_dict={"positions": [[1, 1, 1]], "c": "green", "at": self.instance_id, "r": 20})
        return self.factory.objects_dict

    def send_parameters(self):
        return

    def create(self):
        print(f"Created client n°{self.instance_id}")

    async def animate(self):

        # Generate a random position between [-1, -1, -1] and [1, 1, 1]
        self.input = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1)])

        # The mesh is a sphere of radius 1. We compute the signed distance to the sphere
        # positive = exterior, negative = interior, null = on the surface of the sphere
        self.output = float(np.linalg.norm(self.input) - 1.0)

    async def step(self):  # This function is called by a request of the server
        await self.animate()
        # get_prediction ask the neural network its output for the given input
        neural_network_prediction = await self.get_prediction(input_array=self.input)

        # Specify how to use the prediction
        self.apply_prediction(neural_network_prediction)
        await self.onStep()

    def apply_prediction(self, prediction):
        # Point cloud
        self.factory.updateObject_dict(object_id=1, new_data_dict={'positions': self.input})
        # Ground truth value
        self.factory.updateObject_dict(object_id=2, new_data_dict={'position': np.array([1, self.output, 1])})
        # Prediction value
        self.factory.updateObject_dict(object_id=3, new_data_dict={'position': np.array([1, prediction[0], 1])})

    async def onStep(self):
        # Send visualisation data to update
        await self.update_visualisation(visu_dict=self.factory.updated_object_dict)
        # Send the training data
        await self.__send_training_data(network_input=self.input, network_output=self.output)
