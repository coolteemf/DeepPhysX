"""
ArmadilloPrediction
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
from ArmadilloTraining import ArmadilloTraining, np
from parameters import p_forces, p_model


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

        self.create_model['fem'] = False
        self.visualizer = False

        # Amplitudes pattern
        self.amplitudes = None
        self.idx_amplitude = 0
        # Directions pattern
        self.directions = [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]
        self.idx_direction = 0
        # Zone index
        self.idx_zone = 0

    def recv_parameters(self, param_dict):
        """
        Exploit received parameters before scene creation.
        """

        # Receive visualizer option (either True for Vedo, False for SOFA GUI)
        self.visualizer = param_dict['visualizer'] if 'visualizer' in param_dict else self.visualizer
        step = 0.1 if self.visualizer else 0.02
        self.amplitudes = np.arange(0, 1, step).tolist() + np.arange(1, -1, -step).tolist() + \
                          np.arange(-1, 0, step).tolist()

    def send_visualization(self):
        """
        Define and send the initial visualization data dictionary. Automatically called whn creating Environment.
        """

        # Nothing to visualize if the predictions are run in SOFA GUI.
        if self.visualizer:
            # Add the FEM model (object will have id = 0)
            self.factory.add_object(object_type='Mesh', data_dict={'positions': self.n_visu.position.value.copy(),
                                                                   'cells': self.n_visu.triangles.value.copy(),
                                                                   'at': self.instance_id,
                                                                   'c': 'green'})

            # Arrows representing the force fields (object will have id = 1)
            self.factory.add_object(object_type='Arrows', data_dict={'positions': [0, 0, 0],
                                                                     'vectors': [0., 0., 0.],
                                                                     'c': 'green',
                                                                     'at': self.instance_id})

        return self.factory.objects_dict

    def onAnimateBeginEvent(self, event):
        """
        Called within the Sofa pipeline at the beginning of the time step. Define force vector.
        """

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

        # Update force amplitude and area indices
        if self.idx_amplitude == len(self.amplitudes) - 1 and self.idx_zone == len(self.cff) - 1:
            self.idx_direction = (self.idx_direction + 1) % len(self.directions)
        if self.idx_amplitude == len(self.amplitudes) - 1:
            self.idx_zone = (self.idx_zone + 1) % len(self.cff)
        self.idx_amplitude = (self.idx_amplitude + 1) % len(self.amplitudes)

    def onAnimateEndEvent(self, event):
        """
        Called within the Sofa pipeline at the end of the time step. Compute training data and apply prediction.
        """

        # Compute training data
        input_array = self.compute_input()

        # Send training data
        self.set_training_data(input_array=input_array,
                               output_array=np.array([]))

    def check_sample(self):
        """
        Check if the produced sample is correct. Automatically called by DeepPhysX to check sample validity.
        """

        # See the network prediction in every case.
        return True

    def apply_prediction(self, prediction):
        """
        Apply the predicted displacement to the NN model. Automatically called by DeepPhysX.
        """

        # Reshape to correspond regular grid, transform to sparse grid
        U = np.reshape(prediction, self.data_size)
        self.n_sparse_grid_mo.position.value = self.n_sparse_grid_mo.rest_position.value + U

        # Update visualization if required
        if self.visualizer:
            self.update_visual()

    def update_visual(self):
        """
        Update the visualization data dict.
        """

        # Update visualization data
        self.factory.update_object_dict(object_id=0, new_data_dict={'position': self.n_visu.position.value.copy()})
        position, vector = [], []
        for cff in self.cff:
            position += list(self.n_surface_mo.position.value[cff.indices.value])
            vector += list(0.25 * cff.forces.value / p_model.scale)
        self.factory.update_object_dict(object_id=1, new_data_dict={'positions': position,
                                                                    'vectors': vector})
        # Send updated data
        self.update_visualisation(visu_dict=self.factory.updated_object_dict)
