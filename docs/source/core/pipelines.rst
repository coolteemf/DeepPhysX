CORE - Using a Pipeline
=======================

Dedicated code in ``Core/Pipelines`` repository.

General Policy
--------------

Several *Pipelines* are available with DeepPhysX, allowing the user to:

    * **Generate** synthetic data from simulations → :guilabel:`DataGenerator`
    * **Train** Artificial Neural Networks with the synthetic Dataset → :guilabel:`Trainer`
    * Use the **predictions** of trained Artificial Neural Networks inside the simulations → :guilabel:`Runner`

A *Pipeline* is always associated with a **working session**, whether it already exists or whether it is automatically
created when the *Pipeline* is launched.
A repository containing a training session will contain the following files and folders:

  :``dataset``: This folder contains the partitions of training data, a json file which stores the *Dataset*
                information, and additional optional stuff related to data generation.
  :``network``: This folder contains the parameters of a trained *Network*.
  :``stats``: This folder contains a file readable by Tensorboard which contains the analysis of the evolution of
              the training session.
  :``info.txt``: This file gathers the descriptions of all the involved objects during the working session.
                 The user can also provide more description about the current working session in this file.

The main policy for creating and using a *Pipeline* is to first define a *Configuration* for the components involved by
this *Pipeline*.
*Configurations* are then mandatory for *Dataset*, *Network* and *Environment* if one wants to create a *Pipeline*.
Once these *Configurations* are defined, the *Pipeline* can be created and launched.

.. note::
    More details are provided in dedicated sections.


Pipeline - Data generation
--------------------------

The *DataGenerator* will only involve an *Environment* and a *Dataset*, so this *Pipeline* requires a *Configuration*
for both components to be created.

As the purpose of this *Pipeline* is only to create synthetic data, the working session will always be created at the
same time.
The name of the repository to create can be provided as a parameter.

Furthermore, users have to define which data to save and how much.
For that, the number of intended batches and the size of a batch can be filled.
Users can also choose to only record the input data or the output data.

.. warning::
    This pipeline does not involve a *Network*, the ``get_prediction`` request from Environment is then disabled.
    Trying to launch this request it will lead to an error, so make sure there are no such requests in the data
    production procedure in your *Environment*.

Pipeline - Training
-------------------

The *Trainer* can involve an *Environment*, a *Dataset* and a *Network*, so this *Pipeline* might require a
*Configuration* for these components to be created.

There are several ways to use this pipeline:

    1. Training a *Network* from scratch
    2. Training a *Network* with an existing *Dataset*
    3. Training a *Network* from an existing *Network* state

1. To train a *Network* from scratch, the *Trainer* requires the whole set of *Configurations*.
   A new working session will be created, whose name can be set as a parameter.

2. Training a new *Network* with an existing *Dataset* is considered as creating a new working session.
   The path to the *Dataset* to use has to be provided as a parameter.
   Using an *Environment* is not mandatory since the training data can already have the right format to feed the
   *Network*.
   If some data computation must be performed between the *Dataset* and the *Network*, an *Environment* can be created
   with the specific *Configuration* (see more in the dedicated section).

3. Training from an existing *Network* state can be done both in an existing session or in a new session.
   If you want to work in the same session, you have to configure the *Trainer* to do so, otherwise a new working
   session will be automatically created.
   In the same session, a new set of trained parameters will be added in the ``network`` repository, either trained
   with data from an external *Dataset* (whose path must be provided) or with data from the *Environment* (whose
   *Configuration* must be provided).

The last parameters to set in the Trainer are:

    * The number of *epochs* the training loop should complete during the session
    * The number of *batches* used during a single epoch
    * The number of *samples* in a single batch

.. note::
    By default, the training data will be produced inside the *Environment* during the first epoch and then re-used
    from the *Dataset* for the remaining epochs.
    If you always need to use data from the *Environment*, you can specify this in its *Configuration*.



Pipeline - Prediction
---------------------

The *Runner* always requires a *Network* to compute predictions and an *Environment* to apply them, so this *Pipeline*
will always require the corresponding *Configurations*.

This *Pipeline* always works with an existing working session, no new sessions can be created within a *Runner*.
The path to the session is therefore required, assuming that it contains a trained *Network*.

The *Runner* can either run a specified number of steps or run an infinite loop.

A *Dataset* configuration can be provided.
In this case, the *Runner* can record input or / and output data.
Each sample computed during the prediction phase will then be added to the *Dataset* in dedicated partitions.
