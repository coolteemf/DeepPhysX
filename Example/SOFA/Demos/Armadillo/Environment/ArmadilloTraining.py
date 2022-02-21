"""
ArmadilloTraining
Simulation of an Armadillo with FEM computed simulations.
The SOFA simulation contains two models of an Armadillo :
    * one to apply forces and compute deformations
    * one to apply the network predictions
Training data are produced at each time step :
    * input : applied forces on each surface node
    * output : resulted displacement of each surface node
"""

# Python imports
import os
import sys

# Working session imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ArmadilloSofa import ArmadilloSofa, p_model, np


class ArmadilloTraining(ArmadilloSofa):

    def __init__(self,
                 root_node,
                 ip_address='localhost',
                 port=10000,
                 instance_id=0,
                 number_of_instances=1,
                 as_tcp_ip_client=True,
                 environment_manager=None):

        ArmadilloSofa.__init__(self,
                               root_node=root_node,
                               ip_address=ip_address,
                               port=port,
                               instance_id=instance_id,
                               number_of_instances=number_of_instances,
                               as_tcp_ip_client=as_tcp_ip_client,
                               environment_manager=environment_manager)

        self.create_model['nn'] = True
        self.input_size = (p_model.nb_nodes, 3)
        self.output_size = (p_model.nb_nodes, 3)

    def send_visualization(self):
        """
        Define and send the initial visualization data dictionary. Automatically called whn creating Environment.
        """

        # Add the FEM model
        self.factory.add_object(object_type='Mesh', data_dict={'positions': self.f_visu.position.value.copy(),
                                                               'cells': self.f_visu.triangles.value.copy(),
                                                               'at': self.instance_id,
                                                               'c': 'green'})
        # Add the NN model with translation
        # T = np.array(p_model.vedo_translation * self.n_visu.position.shape[0]).reshape(self.n_visu.position.shape)
        # self.factory.add_object(object_type='Mesh', data_dict={'position': self.n_visu.position.value.copy() + T,
        #                                                        'cells': self.n_visu.triangles.value.copy(),
        #                                                        'at': self.instance_id,
        #                                                        'c': 'orange'})
        # Return the initial visualization data
        return self.factory.objects_dict

    def onAnimateEndEvent(self, event):
        """
        Called within the Sofa pipeline at the end of the time step. Compute training data and apply prediction.
        """

        # Compute training data
        input_array = self.compute_input()
        output_array = self.compute_output()

        # Get and apply the prediction of the network
        self.apply_prediction(self.get_prediction(input_array=input_array))

        # Send training data
        self.set_training_data(input_array=input_array,
                               output_array=output_array)

    def compute_input(self):
        """
        Compute force vector for the whole surface.
        """

        F = np.zeros(self.input_size)
        # Add each force vector to the network input
        for force_field in self.cff:
            f = force_field.forces.value.copy()
            idx = force_field.indices.value.copy()
            F[idx] = f
        return F.copy() / p_model.scale

    def compute_output(self):
        """
        Compute displacement vector for the whole surface.
        """

        # Compute generated displacement
        U = self.f_surface_mo.position.value - self.f_surface_mo.rest_position.value
        return U.copy() * 10

    def apply_prediction(self, prediction):
        """
        Apply the predicted displacement to the NN model, update visualization data.
        """

        # Reshape to correspond regular grid, transform to sparse grid
        U = np.reshape(prediction, self.output_size) / 10
        self.n_surface_mo.position.value = self.n_surface_mo.rest_position.array() + U

        # Update visualization data
        self.factory.update_object_dict(object_id=0, new_data_dict={'position': self.f_visu.position.value.copy()})
        # Update the NN model with translation
        # translation = np.array(p_model.vedo_translation * self.n_visu.position.value.shape[0]).reshape(
        #     self.n_visu.position.value.shape)
        # self.factory.update_object_dict(object_id=1, new_data_dict={'position': self.n_visu.position.value.copy() +
        #                                                                         translation})
        # Send updated data
        self.update_visualisation(visu_dict=self.factory.updated_object_dict)
