## Features review

This repository contains a full review of DeepPhysX features.

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


* 
* `GradienDescent.py`: A funny example to visualize the gradient descent process
