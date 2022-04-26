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
from ArmadilloTraining import ArmadilloTraining, p_model, np
from parameters import p_forces, p_grid


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

        # Amplitudes pattern
        step = 0.2
        self.amplitudes = np.arange(0, 1, step).tolist() + np.arange(1, -1, -step).tolist() + \
                          np.arange(-1, 0, step).tolist()
        self.idx_amplitude = 0
        # Directions pattern
        self.directions = [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]
        self.idx_direction = 0
        # Zone index
        self.idx_zone = 0

    # def send_visualization(self):
    #     """
    #     Define and send the initial visualization data dictionary. Automatically called whn creating Environment.
    #     """
    #
    #     # Nothing to visualize since the predictions are run in SOFA GUI.
    #     return self.factory.objects_dict

    def send_visualization(self):
        """
        Define and send the initial visualization data dictionary. Automatically called when creating Environment.
        """

        self.grid = [[p_grid.bbox_anchor[0] + i * p_grid.bbox_size[0] / p_grid.nb_cells[0] for i in
                      range(p_grid.nb_cells[0] + 1)],
                     [p_grid.bbox_anchor[1] + i * p_grid.bbox_size[1] / p_grid.nb_cells[1] for i in
                      range(p_grid.nb_cells[1] + 1)],
                     [p_grid.bbox_anchor[2] + i * p_grid.bbox_size[2] / p_grid.nb_cells[2] for i in
                      range(p_grid.nb_cells[2] + 1)]]
        self.grid_positions = []
        for z in self.grid[2]:
            for y in self.grid[1]:
                for x in self.grid[0]:
                    self.grid_positions.append(np.array([x, y, z]))
        self.grid_positions = np.array(self.grid_positions)

        # Add the FEM model
        self.factory.add_object(object_type='Mesh', data_dict={'positions': self.n_visu.position.value.copy(),
                                                               'cells': self.n_visu.triangles.value.copy(),
                                                               'at': self.instance_id,
                                                               'c': 'green'})

        # Arrows representing the force fields (object will have id = 1)
        self.factory.add_object(object_type='Arrows',
                                data_dict={'positions': [0, 0, 0],
                                           'vectors': [0., 0., 0.],
                                           'c': 'green',
                                           'at': self.instance_id})

        # Grid
        self.factory.add_object(object_type='Points',
                                data_dict={'positions': self.n_sparse_grid_mo.position.value,
                                           'r': 2,
                                           'c': 'grey',
                                           'at': self.instance_id})

        # Prediction
        self.factory.add_object(object_type='Arrows',
                                data_dict={'positions': [0, 0, 0],
                                           'vectors': [0., 0., 0.],
                                           'c': 'red',
                                           'at': self.instance_id})

        # Return the initial visualization data
        return self.factory.objects_dict

    def onAnimateBeginEvent(self, event):
        """
        Called within the Sofa pipeline at the beginning of the time step. Define force vector.
        """

        # Reset positions
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

    def apply_prediction(self, prediction):
        """
        Apply the predicted displacement to the NN model.
        """

        # Reshape to correspond regular grid, transform to sparse grid
        U = np.reshape(prediction, self.data_size)
        U_sparse = U[self.idx_sparse_to_regular]
        self.n_sparse_grid_mo.position.value = self.n_sparse_grid_mo.rest_position.array() + U_sparse


        # # Update visualization data
        # self.factory.update_object_dict(object_id=0, new_data_dict={'position': self.n_visu.position.value.copy()})

        # position, vec = [], []
        # for cff in self.cff:
        #     if list(cff.force.value) != [0., 0., 0.]:
        #         position += list(self.n_surface_mo.position.value[cff.indices.value])
        #         vec += list(0.25 * cff.force.value / p_model.scale)
        # if len(position) == 0 or len(vec) == 0:
        #     position, vec = [[0., 0., 0.]], [[0., 0., 0.]]
        # self.factory.update_object_dict(object_id=1,
        #                                 new_data_dict={'positions': position,
        #                                                'vectors': vec})
        #
        # self.factory.update_object_dict(object_id=2,
        #                                 new_data_dict={'positions': self.n_sparse_grid_mo.position.value})
        #
        # self.factory.update_object_dict(object_id=3,
        #                                 new_data_dict={'positions': self.n_sparse_grid_mo.position.value,
        #                                                'vectors': U_sparse})

        # # Send updated data
        # self.update_visualisation(visu_dict=self.factory.updated_object_dict)

    def check_sample(self):
        """
        Check if the produced sample is correct. Automatically called by DeepPhysX to check sample validity.
        """

        # See the network prediction even if the solver diverged.
        return True
