"""
BeamPrediction
Simulation of a Beam with NN computed simulations.
The SOFA simulation contains the model used to apply the network predictions.
Training data are produced at each time step :
    * input : applied forces on each surface node
    * output : resulted displacement of each surface node
"""

# Python imports
import os
import sys

# Working session imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from BeamTraining import BeamTraining, p_grid, np


class BeamPrediction(BeamTraining):

    def __init__(self,
                 root_node,
                 ip_address='localhost',
                 port=10000,
                 instance_id=0,
                 number_of_instances=1,
                 as_tcp_ip_client=True,
                 environment_manager=None):

        BeamTraining.__init__(self,
                              root_node=root_node,
                              ip_address=ip_address,
                              port=port,
                              instance_id=instance_id,
                              number_of_instances=number_of_instances,
                              as_tcp_ip_client=as_tcp_ip_client,
                              environment_manager=environment_manager)

        self.create_model['fem'] = False
        self.range = np.concatenate((np.arange(0, 1, 0.01),
                                     np.arange(1, -1, -0.01),
                                     np.arange(-1, 0, 0.01)))
        self.idx_range = 0
        self.force_value = None
        self.indices_value = None

    def send_visualization(self):
        """
        Define and send the initial visualization data dictionary. Automatically called whn creating Environment.
        """

        # Nothing to visualize since the predictions are run in SOFA GUI.
        return self.factory.objects_dict

    def onAnimateBeginEvent(self, event):
        """
        Called within the Sofa pipeline at the beginning of the time step. Define force vector.
        """

        # Reset positions
        self.n_grid_mo.position.value = self.n_grid_mo.rest_position.value

        # Reset force amplitude index
        if self.idx_range == len(self.range):
            self.idx_range = 0

        # Create a random box ROI, select nodes of the surface
        if self.idx_range == 0:
            indices = []
            # Avoid empty box
            while len(indices) == 0:
                # Define random box
                x_min = np.random.randint(p_grid.min[0], p_grid.max[0] - 10)
                x_max = np.random.randint(x_min + 10, p_grid.max[0])
                y_min = np.random.randint(p_grid.min[1], p_grid.max[1] - 10)
                y_max = np.random.randint(y_min + 10, p_grid.max[1])
                z_min = np.random.randint(p_grid.min[2], p_grid.max[2] - 10)
                z_max = np.random.randint(z_min + 10, p_grid.max[2])
                # Set the new bounding box
                self.root.nn.removeObject(self.cff_box)
                self.cff_box = self.root.nn.addObject('BoxROI', name='ForceBox', drawBoxes=False, drawSize=1,
                                                      box=[x_min, y_min, z_min, x_max, y_max, z_max])
                self.cff_box.init()
                # Get the intersection with the surface
                indices = list(self.cff_box.indices.value)
                indices = list(set(indices).intersection(set(self.idx_surface)))
            # Create a random force vector
            F = 15 * np.random.uniform(low=-1, high=1, size=(3,))
            # Keep value
            self.force_value = F
            self.indices_value = indices

        # Update force field
        F = self.range[self.idx_range] * self.force_value
        self.root.nn.removeObject(self.cff)
        self.cff = self.root.nn.addObject('ConstantForceField', name='CFF', showArrowSize=0.5,
                                          indices=self.indices_value,
                                          force=list(F))
        self.cff.init()
        self.idx_range += 1

    def onAnimateEndEvent(self, event):
        """
        Called within the Sofa pipeline at the end of the time step. Compute training data and apply prediction.
        """

        # Get a prediction and apply it on NN model
        input_array = self.compute_input()

        # Send training data
        self.set_training_data(input_array=input_array,
                               output_array=np.array([]))

    def check_sample(self):
        """
        Check if the produced sample is correct. Automatically called by DeepPhysX to check sample validity.
        """

        # See the network prediction even if the solver diverged.
        return True
