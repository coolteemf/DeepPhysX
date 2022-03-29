"""
ArmadilloValidation
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
from ArmadilloTraining import ArmadilloTraining, p_model, np


class ArmadilloValidation(ArmadilloTraining):

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

        self.l2_error, self.MSE_error = [], []
        self.l2_deformation, self.MSE_deformation = [], []

        self.compute_sample = True

    def recv_parameters(self, param_dict):
        """
        Exploit received parameters before scene creation.
        """

        self.compute_sample = param_dict['compute_sample'] if 'compute_sample' in param_dict else self.compute_sample

    def createFEM(self):
        """
        FEM model of Armadillo. Used to apply forces and compute deformations.
        """

        if self.compute_sample:
            ArmadilloTraining.createFEM(self)

        else:
            # Create child node
            self.root.addChild('fem')

            # Surface
            self.root.fem.addChild('surface')
            self.f_surface_topo = self.root.fem.surface.addObject('TriangleSetTopologyContainer', name='SurfaceTopo',
                                                                  src='@../../MeshCoarse')
            self.f_surface_mo = self.root.fem.surface.addObject('MechanicalObject', name='SurfaceMO',
                                                                src='@../../MeshCoarse', showObject=False)

            # Fixed section
            self.root.fem.surface.addObject('BoxROI', name='FixedBox', box=p_model.fixed_box, drawBoxes=True,
                                            drawSize=1.)
            self.root.fem.surface.addObject('FixedConstraint', indices='@FixedBox.indices')

            # Forces
            self.create_forces()

            # Visual
            self.root.fem.addChild('visual')
            self.f_visu = self.root.fem.visual.addObject('OglModel', name="OGL", src='@../../MeshCoarse', color='green')
            self.root.fem.visual.addObject('BarycentricMapping', input='@../surface/SurfaceMO', output='@./')

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

        if self.compute_sample:
            ArmadilloTraining.onAnimateBeginEvent(self, event)

        else:
            # Reset positions
            if self.create_model['fem']:
                self.f_surface_mo.position.value = self.f_surface_mo.rest_position.value
            if self.create_model['nn']:
                self.n_surface_mo.position.value = self.n_surface_mo.rest_position.value

    def onAnimateEndEvent(self, event):
        """
        Called within the Sofa pipeline at the end of the time step. Compute training data and apply prediction.
        """

        if self.sample_in is not None:
            input_array = self.sample_in
            output_array = self.sample_out
        else:
            input_array = self.compute_input()
            output_array = self.compute_output()

        if self.sample_out is not None:
            U = np.reshape(self.sample_out, self.data_size) * p_model.size
            self.f_surface_mo.position.value = self.f_surface_mo.rest_position.value + U

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
        self.compute_metrics()

    def check_sample(self):
        """
        Check if the produced sample is correct. Automatically called by DeepPhysX to check sample validity.
        """

        # See the network prediction even if the solver diverged.
        return True

    def compute_metrics(self):
        """
        Compute L2 error and MSE for each sample.
        """

        pred = self.n_surface_mo.position.value - self.n_surface_mo.rest_position.value
        gt = self.f_surface_mo.position.value - self.f_surface_mo.rest_position.value

        # Compute metrics only for non-zero displacements
        if np.linalg.norm(gt) != 0.:
            error = (gt - pred).reshape(-1)
            self.l2_error.append(np.linalg.norm(error))
            self.MSE_error.append((error.T @ error) / error.shape[0])
            self.l2_deformation.append(np.linalg.norm(gt))
            self.MSE_deformation.append((gt.reshape(-1).T @ gt.reshape(-1)) / gt.shape[0])

    def close(self):

        if len(self.l2_error) > 0:
            print("\nL2 ERROR Statistics :")
            print(f"\t- Distribution : {1e3 * np.mean(self.l2_error)} ± {1e3 * np.std(self.l2_error)} mm")
            print(f"\t- Extrema : {1e3 * np.min(self.l2_error)} -> {1e3 * np.max(self.l2_error)} mm")
            relative_error = np.array(self.l2_error) / np.array(self.l2_deformation)
            print(f"\t- Relative Distribution : {1e2 * relative_error.mean()} ± {1e2 * relative_error.std()} %")
            print(f"\t- Relative Extrema : {1e2 * relative_error.min()} -> {1e2 * relative_error.max()} %")

            print("\nMSE Statistics :")
            print(f"\t- Distribution : {1e6 * np.mean(self.MSE_error)} ± {1e6 * np.std(self.MSE_error)} mm²")
            print(f"\t- Extrema : {1e6 * np.min(self.MSE_error)} -> {1e6 * np.max(self.MSE_error)} mm²")
            relative_error = np.array(self.MSE_error) / np.array(self.MSE_deformation)
            print(f"\t- Relative Distribution : {1e2 * relative_error.mean()} ± {1e2 * relative_error.std()} %")
            print(f"\t- Relative Extrema : {1e2 * relative_error.min()} -> {1e2 * relative_error.max()} %")
