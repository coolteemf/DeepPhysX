"""
prediction.py
Run the pipeline BaseRunner to check the predictions of the trained network.
"""

# Python related imports
import os

# DeepPhysX related imports
from DeepPhysX_Core.Pipelines.BaseRunner import BaseRunner
from DeepPhysX_Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX_PyTorch.FC.FCConfig import FCConfig
from DeepPhysX_Core.Visualizer.VedoVisualizer import VedoVisualizer

# Session imports
from Environment import MeanEnvironment


def launch_prediction(session_path):
    # Define the number of points and the dimension
    nb_points = 30
    dimension = 3
    # Environment configuration
    environment_config = BaseEnvironmentConfig(environment_class=MeanEnvironment,
                                               visualizer=VedoVisualizer,
                                               param_dict={'constant': False,
                                                           'data_size': [nb_points, dimension],
                                                           'sleep': False,
                                                           'allow_requests': False},
                                               as_tcp_ip_client=False)
    # Fully Connected configuration (the number of neurones on the first and last layer is defined by the total amount
    # of parameters in the input and the output vectors respectively)
    network_config = FCConfig(dim_layers=[nb_points * dimension, nb_points * dimension, dimension],
                              dim_output=dimension)
    # Dataset configuration with the path to the existing Dataset
    # Create DataGenerator
    trainer = BaseRunner(session_dir=os.path.join(os.getcwd(), 'sessions/online_training'),
                         environment_config=environment_config,
                         network_config=network_config)
    # Launch the training session
    trainer.execute()


if __name__ == '__main__':

    is_offline_session = os.path.exists(os.path.join(os.getcwd(), 'sessions/offline_training'))
    is_online_session = os.path.exists(os.path.join(os.getcwd(), 'sessions/online_training'))

    if not is_online_session and not is_offline_session:
        print("Trained Network required, 'sessions/online_training' or 'sessions/offline_training' not found. "
              "Run onlineTraining.py script first.")
        from onlineTraining import launch_training
        launch_training()
        session_dir = 'sessions/online_training'
    else:
        session_dir = 'sessions/online_training' if is_online_session else 'sessions/offline_training'

    launch_prediction(session_dir)
