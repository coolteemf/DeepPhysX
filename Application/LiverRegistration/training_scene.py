import os
import numpy as np

from LiverEnvironmentConfig import LiverEnvironmentConfig
from utils import compute_grid_resolution

# ENVIRONMENT PARAMETERS
# Liver variables
mesh_folder = os.path.dirname(os.path.realpath(__file__))
mesh_file = mesh_folder + "/liver.obj"
translation = [-0.0134716, 0.021525, -0.427]
fixed_point = np.array([-0.00338, -0.0256, 0.52]) + np.array(translation)
fixed_width = np.array([0.07, 0.05, 0.04])
fixed_box = (fixed_point - fixed_width / 2.).tolist() + (fixed_point + fixed_width / 2.).tolist()
camera_position = np.array([-0.177458, 0.232606, 0.780813])
p_liver = {'mesh_file': mesh_file,
           'translation': translation,
           'camera_position': camera_position,
           'fixed_box': fixed_box,
           'fixed_point': fixed_point}
# Grid variables
cell_size = 0.07
margins = np.array([0.02, 0.02, 0.02])
min_bbox = np.array([-0.130815, -0.107192, 0.00732511]) - margins
max_bbox = np.array([0.0544588, 0.0967464, 0.15144]) + margins
grid_resolution = compute_grid_resolution(max_bbox, min_bbox, cell_size)
print("Grid resolution is {}".format(grid_resolution))
bbox_size = max_bbox - min_bbox
max_bbox = max_bbox.tolist()
min_bbox = min_bbox.tolist()
nb_cells_x = grid_resolution[0] - 1
nb_cells_y = grid_resolution[1] - 1
nb_cells_z = grid_resolution[2] - 1
p_grid = {'min_bbox': min_bbox,
          'max_bbox': max_bbox,
          'bbox_size': bbox_size,
          'nb_cells': [nb_cells_x, nb_cells_y, nb_cells_z],
          'grid_resolution': grid_resolution}
# Forces variables
nb_simultaneous_forces = 20
amplitude_scale = 0.05
inter_distance_thresh = 0.06
p_forces = {'nb_simultaneous_forces': nb_simultaneous_forces,
            'amplitude_scale': amplitude_scale,
            'inter_distance_thresh': inter_distance_thresh}

# TRAINING PARAMETERS
nb_epoch = 100
nb_batch = 30
batch_size = 5


def createScene(root_node):
    # Environment config
    env_config = LiverEnvironmentConfig(root_node=root_node,
                                        p_liver=p_liver,
                                        p_grid=p_grid,
                                        p_forces=p_forces)
    root_node = env_config.createEnvironment()

    """# Network config
    net_config = UNetConfig(network_name="liver_test",
                            loss=mse_loss,
                            lr=1e-4,
                            optimizer=Adam,
                            save_each_epoch=True,
                            steps=3,
                            first_layer_channels=128,
                            nb_classes=3,
                            nb_input_channels=1,
                            two_sublayers=True,
                            nb_dims=3,
                            border_mode='same')

    # Dataset config
    dataset_config = BaseDatasetConfig(partition_size=0.2,
                                       generate_data=True,
                                       shuffle_dataset=True)

    # Here: Trainer
    trainer = BaseTrainer(session_name="liver_test",
                          nb_epochs=nb_epoch,
                          nb_batches=nb_batch,
                          batch_size=batch_size,
                          network_config=net_config,
                          dataset_config=dataset_config,
                          environment_config=env_config)
    trainer.execute()"""


if __name__ == '__main__':
    print("Go through main function")
    createScene(None)
