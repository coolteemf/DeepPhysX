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

# Python related imports
import os
import sys

# Session related imports
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
        self.data_size = (p_model.nb_nodes, 3)
        self.is_network = True

    def recv_parameters(self, param_dict):
        """
        Exploit received parameters before scene creation.
        """

        self.is_network = param_dict['is_network'] if 'is_network' in param_dict else self.is_network

    def send_visualization(self):
        """
        Define and send the initial visualization data dictionary. Automatically called when creating Environment.
        """

        # Add the FEM model
        self.factory.add_object(object_type='Mesh', data_dict={'positions': self.f_visu.position.value.copy(),
                                                               'cells': self.f_visu.triangles.value.copy(),
                                                               'at': self.instance_id,
                                                               'c': 'green'})
        # Return the initial visualization data
        return self.factory.objects_dict

    def onAnimateEndEvent(self, event):
        """
        Called within the Sofa pipeline at the end of the time step. Compute training data and apply prediction.
        """
        # Compute training data
        input_array = self.compute_input()
        output_array = self.compute_output()
        self.update_visualization()

        # Send training data
        self.set_training_data(input_array=input_array,
                               output_array=output_array)

    def compute_input(self):
        """
        Compute force vector for the whole surface.
        """

        # Init encoded force vector to zero
        F = np.zeros(self.data_size, dtype=np.double)

        # Encode each force field
        for force_field in self.cff:
            for i in force_field.indices.value:
                # Get the lis of nodes composing a cell containing a point from the force field
                p = self.f_surface_mo.position.value[i]
                cell = self.regular_grid.cell_index_containing(p)
                # For each node of the cell, encode the force value
                for node in self.regular_grid.node_indices_of(cell):
                    if node < self.nb_nodes_regular_grid and np.linalg.norm(F[node]) == 0.:
                        F[node] = force_field.force.value
        return F.copy()

    def compute_output(self):
        """
        Compute displacement vector for the whole surface.
        """

        # Write the position of each point from the sparse grid to the regular grid
        actual_positions_on_regular_grid = np.zeros(self.data_size, dtype=np.double)
        actual_positions_on_regular_grid[self.idx_sparse_to_regular] = self.f_sparse_grid_mo.position.array()
        return np.subtract(actual_positions_on_regular_grid, self.regular_grid_rest_shape).copy()

    def apply_prediction(self, prediction):
        """
        Apply the predicted displacement to the NN model, update visualization data.
        """

        # Reshape to correspond regular grid, transform to sparse grid
        U = np.reshape(prediction, self.data_size)
        U_sparse = U[self.idx_sparse_to_regular]
        self.n_sparse_grid_mo.position.value = self.n_sparse_grid_mo.rest_position.array() + U_sparse
        self.update_visualization()

    def update_visualization(self):
        # Update visualization data
        self.factory.update_object_dict(object_id=0, new_data_dict={'position': self.f_visu.position.value.copy()})
        # Send updated data
        self.update_visualisation(visu_dict=self.factory.updated_object_dict)
