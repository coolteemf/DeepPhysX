## Features review

This repository contains examples focused on training a Fully Connected Network 
to predict the mean value of a vector. These examples illustrate the main features
of DeepPhysX framework.

**Warning**: This working session uses the Fully Connected implementation from 
DeepPhysX_Torch package, please make sure it is installed if you want to run
training scripts. Otherwise, you can download training data and / or trained
network by running ___.

###Content

Examples of an Environment: how to create it, how to send and receive data, 
how to initialize and update visualization data, how to send requests.
* `Environment.py`: Implementation of the Data producer
* `EnvironmentOffscreen.py`: Same implementation as Environment without visualization

Examples of Dataset generation: how to create the pipeline, how to use multiprocessing 
to speed up the data production.
* `DataGeneration.py`: Launch the data generation pipeline.
* `Multiprocessing.py`: Compare data production between single process and multiprocess.

Examples of Training sessions: how to create and launch the pipeline.
* `OfflineTraining.py`: Train the Network with an existing Dataset.
* `OnlineTraining.py`: Train the Network while producing the Dataset simultaneously and visualizing predictions.

Examples of Prediction sessions: how to create and launch the pipeline.
* `Prediction.py`: Run predictions from trained Network in the Environment.
* `GradienDescent.py`: A funny example to visualize the gradient descent process
