Overview
========

Packages
--------

The DeepPhysX project is divided into several python packages: a Core package which is able to communicate with a
simulation package and a learning package.
This way, the Core has no dependencies neither to a simulation framework nor an AI framework, and any of those
frameworks could be compatible with DeepPhysX.

*Add the packages overview doc*

Core
""""

This package rules the communications and the data flow between the AI side and the simulation side, the storage of the
Dataset, the visualization tools.
This package is named :guilabel:`DeepPhysX_Core`.

.. admonition:: Dependencies

    NumPy, Tensorboard, Vedo

Simulation
""""""""""

This package provides a DeepPhysX compatible API for your simulations.
For DeepPhysX, each simulation package is written for a particular simulation framework (written in python or providing
python bindings).
Thus, each package will be named :guilabel:`DeepPhysX_` + :guilabel:`Simulation_framework_name`.

.. admonition:: Available simulation packages

    :guilabel:`DeepPhysX_Sofa` designed for `SOFA <https://www.sofa-framework.org/>`_

Learning
""""""""

This package provides a DeepPhysX compatible API for your Deep Learning Algorithms.
In the same way, each learning package is written for a particular AI python framework.
Thus, each package will be named :guilabel:`DeepPhysX_` + :guilabel:`AI_framework_name`.

.. admonition:: Available learning packages

    :guilabel:`DeepPhysX_Torch` designed for `PyTorch <https://pytorch.org/>`_


Architecture
------------

This section describes both the links between the components of Core package and the links between Core package and
the other packages.

Users might use one of the provided pipelines for their **data generation**, their **training session** or their
**predictions**.
These pipelines triggers a **loop** which defines the number of samples to produce, the number of epochs needed for the
training session or the number of steps of prediction.

*Add the project overview doc*

The pipeline will involve several components (some producing data, some consuming data), but the pipeline will
communicate with their **Managers** first.
A main **Manager** will provide the pipeline an intermediary with all the existing managers:

* The Environment Manager will manage the Environment component to create it, to trigger steps of simulations, to
  produce synthetic training data, to provide predictions of the network if required, and to finally shutdown the
  Environment.

  .. note::
    This Manager can communicate directly with a single Environment or with a Server which shares information with
    several Environments in parallel launched as Client through a custom TCP-IP protocol (see Environment section).

* The Dataset Manager will manage the Dataset component to create storage partitions, to fill these partitions with the
  synthetic training data produced by the Environment and to reload an existing Dataset for training sessions.

  .. note::
    If you choose to generate data and train your network simultaneously (by default for the training pipeline), you
    can generate the Dataset only during the first epoch and then reload this Dataset for the remaining epochs.

  .. note::
    The two above Managers are managed by the Data Manager since both the Environment and the Dataset component can
    provide training data for the network.
    This Data Manager is the one who decides if this data should be requested from the Environment or from the
    Dataset depending on your pipeline configuration.

* The Network Manager will manage several objects to train your neural network:

    * The Network to produce a prediction from an input, to save a set of parameters or to reload a trained network.

    * The Optimizer to compute the loss function and to optimize the parameters of the Network. This component uses
      existing loss functions and optimizers in the chosen AI framework.

    * The Data Transformation to convert the type of training data sent from Environment to a compatible type for the
      AI framework you use and vice versa, to transform training data before a prediction, before the loss computation
      and before sending the prediction to the Environment.

  .. note::
    You can define your own learning algorithm if the stuff provided in the learning packages is not enough for you.
    You are free to define your own Network architecture, to create your custom loss or optimizer to feed the
    Optimizer, and to compute your required tensor transformations in the Data Transformation component.

* The Visualizer Manager which manages the Visualizer to gather the simulated objects you want to render.
  Factories are provided to create a wide variety of objects (meshes, point clouds, markers, etc).

  .. note::
    You have to specify in your Environment which object you want to create and when you want to update them in the
    rendering window.
    In the case where you run several Environments in parallel, the rendering windows will be split in several
    sub-windows to gather the rendering of your simulations.

* The Stats Manager which manages the analysis of the evolution of a training session.
  These analytical data will be saved as a file readable by Tensorboard.

  .. note::
    Common curves will be automatically provided in the board (such as the evolution of the loss value, the smoothed
    mean and the variance of this loss value per batch and per epoch), but you can add and fill other custom fields
    as well.

.. warning::
    If you try to use the default Network or Environment provided in the Core package, you will quickly see that they
    are not implemented at all.
    The reason is that you need to choose an AI and a simulation python framework to implement them.
    The aim of DeepPhysX learning and simulation packages is to provide a compatible implementation both for DeepPhysX
    and both for the AI and the simulation framework.

.. admonition:: Example

    If you choose PyTorch as your AI framework, you can use or implement a TorchNetwork which inherits from DeepPhysX
    Network and from Torch.nn.module (available in :guilabel:`DeepPhysX_Torch`).
    If you choose SOFA as your simulation framework, you can implement a SofaEnvironment which inherits from DeepPhysX
    Environment and from Sofa.Core.Controller (available in :guilabel:`DeepPhysX_Sofa`).
