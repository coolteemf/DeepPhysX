"""
ArmadilloNetwork
Simulation of an Armadillo with NN computed simulations.
The SOFA simulation contains the models used to apply the network predictions.
Training data are produced at each time step :
    * input : applied forces on each surface node
    * output : resulted displacement of each surface node
"""

# Python imports
import os
import sys

# Working session imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ArmadilloPrediction import ArmadilloPrediction, np
from parameters import p_forces


class ArmadilloNetwork(ArmadilloPrediction):

    def __init__(self,
                 root_node,
                 ip_address='localhost',
                 port=10000,
                 instance_id=0,
                 number_of_instances=1,
                 as_tcp_ip_client=True,
                 environment_manager=None):

        ArmadilloPrediction.__init__(self,
                                     root_node=root_node,
                                     ip_address=ip_address,
                                     port=port,
                                     instance_id=instance_id,
                                     number_of_instances=number_of_instances,
                                     as_tcp_ip_client=as_tcp_ip_client,
                                     environment_manager=environment_manager)

        self.create_model['fem'] = False

        # Amplitudes pattern
        step = 0.02
        self.amplitudes = np.arange(0, 1, step).tolist() + np.arange(1, -1, -step).tolist() + \
                          np.arange(-1, 0, step).tolist()
        self.idx_amplitude = 0
        # Directions pattern
        self.directions = [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]
        self.idx_direction = 0
        # Zone index
        self.idx_zone = 0

    def onAnimateBeginEvent(self, event):
        """
        Called within the Sofa pipeline at the beginning of the time step. Define force vector.
        """

        # Reset positions
        if self.create_model['nn']:
            self.n_surface_mo.position.value = self.n_surface_mo.rest_position.value

        # Reset forces
        for cff in self.cff:
            cff.force.value = np.array([0., 0., 0.])

        # Generate new forces
        zone = p_forces.zones[self.idx_zone]
        cff = self.cff[self.idx_zone]
        amplitude = p_forces.amplitude[zone]
        f = np.array([0., 0., 0.])
        for direction in self.directions[self.idx_direction]:
            f[direction] = self.amplitudes[self.idx_amplitude] * amplitude
        cff.force.value = f
        cff.showArrowSize.value = 10 if self.idx_zone == 0 else 100

        if self.idx_amplitude == len(self.amplitudes) - 1 and self.idx_zone == len(self.cff) - 1:
            self.idx_direction = (self.idx_direction + 1) % len(self.directions)

        if self.idx_amplitude == len(self.amplitudes) - 1:
            self.idx_zone = (self.idx_zone + 1) % len(self.cff)

        self.idx_amplitude = (self.idx_amplitude + 1) % len(self.amplitudes)

    def onAnimateEndEvent(self, event):
        """
        Called within the Sofa pipeline at the end of the time step. Compute training data and apply prediction.
        """

        # Get a prediction and apply it on NN model
        input_array = self.compute_input()

        # Send training data
        self.set_training_data(input_array=input_array,
                               output_array=np.array([]))
