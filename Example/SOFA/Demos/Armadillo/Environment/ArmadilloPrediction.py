"""
ArmadilloPrediction
Simulation of an Armadillo with FEM computed simulations.
The SOFA simulation contains two models of an Armadillo :
    * one to apply forces and compute deformations
    * one to apply the network predictions
Training data are produced at each time step :
    * input : applied forces on each surface node
    * output : resulted displacement of each surface node
"""

# Python related imports
import os
import sys

# Session related imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ArmadilloTraining import ArmadilloTraining, np, p_model


class ArmadilloPrediction(ArmadilloTraining):

    def __init__(self,
                 root_node,
                 ip_address='localhost',
                 port=10000,
                 instance_id=0,
                 number_of_instances=1,
                 as_tcp_ip_client=True,
                 environment_manager=None):

        ArmadilloTraining.__init__(self,
                                   root_node=root_node,
                                   ip_address=ip_address,
                                   port=port,
                                   instance_id=instance_id,
                                   number_of_instances=number_of_instances,
                                   as_tcp_ip_client=as_tcp_ip_client,
                                   environment_manager=environment_manager)

    def send_visualization(self):
        """
        Define and send the initial visualization data dictionary. Automatically called whn creating Environment.
        """

        # Nothing to visualize since the predictions are run in SOFA GUI.
        return self.factory.objects_dict

    def onAnimateEndEvent(self, event):
        """
        Called within the Sofa pipeline at the end of the time step. Compute training data and apply prediction.
        """

        # Get a prediction and apply it on NN model
        input_array = self.compute_input()
        output_array = self.compute_output()

        # Send training data
        self.set_training_data(input_array=input_array,
                               output_array=output_array)

    def apply_prediction(self, prediction):
        """
        Apply the predicted displacement to the NN model.
        """

        # Reshape to correspond regular grid, transform to sparse grid
        U = np.reshape(prediction, self.data_size) * p_model.size
        self.n_surface_mo.position.value = self.n_surface_mo.rest_position.array() + U

    def check_sample(self, check_input=True, check_output=True):
        """
        Check if the produced sample is correct. Automatically called by DeepPhysX to check sample validity.
        """

        # See the network prediction even if the solver diverged.
        return True
