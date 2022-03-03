CORE - Using a Network
======================

Dedicated code in ``Core/Network`` repository.

Network Implementation
----------------------

A *Network* cannot be directly implemented from *BaseNetwork* because there is no dependency on any Python AI package.
Even if a *BaseNetwork* is not usable, a dedicated DeepPhysX AI package provides the same API.

.. note::
    Before writing code to create a new *Network* architecture, users should check DeepPhysX AI packages if it is not
    already implemented.

| **Specific**
| The only methods requiring a user implementation are:

:``__init__``: The architecture must be defined when creating the *Network*.

:``forward``: The forward pass of the *Network* must also be defined.

| **Shared**
| Given a Python AI framework, most of the methods should be already implemented if a DeepPhysX package is already
| written for it.
| Otherwise, this list of methods must be implemented:

:``set_train``: Configure the *Network* to training mode (chain rule for gradient will be calculated).

:``set_eval``: Configure the *Network* to prediction mode (no gradient will be calculated).

:``set_device``: Define whether tensors computations are done on the CPU or on the GPU.

:``load_parameters``: Load a set of saved parameters from a file.
                      This set of parameters must be compatible with the current *Network* architecture.

:``get_parameters``: Return the current state of *Network* parameters.

:``save_parameters``: Save the current set of parameters to a file.

:``nb_parameters``: Return the number of parameters in the architecture.

:``transform_from_numpy``: Convert a Numpy array to a tensor with the compatible type.
                           Received data from Core will always be Numpy arrays.

:``transform_to_numpy``: Convert a tensor to a Numpy array.
                         Data provided to Core must be converted to Numpy arrays.


Optimization Implementation
---------------------------

The purpose of the *Optimization* component is to compute the loss associated with the output of the *Network* and to
optimize the parameters considering its gradient.
Same as *BaseNetwork*, *BaseOptimization* cannot be used in training sessions as it must depend on a Python AI
framework.
However, *Optimization* components still share the same API in every DeepPhysX package.
Given a Python AI framework, methods should all be already implemented so that users can use them as it is.

The first methods allow defining the loss function and the optimizer.
These methods should not be overwritten by users:

:``set_loss``: Initialize the loss function.

:``set_optimizer``: Initialize the *Optimizer* with the parameters of the *Network*.

Other methods define the loss computation and the optimization process.
Users can use the default implementation or customize their own processes:

:``compute_loss``: Compute the loss with the defined loss function from prediction and ground truth.

:``transform_loss``: Apply a transformation on the loss value using potential additional data.
                     Additional loss data can be sent from *Environment* (see dedicated section).

:``optimize``: Run an optimization step to adapt the parameters of the *Network* according to the loss gradient.


Data Transformation Implementation
----------------------------------

*DataTransformation* objects are dedicated to tensor transformations at different levels when streaming data through
*Network* and *Optimization* objects.
These transformations are automatically called with the pipeline and apply the identity transformation by default.
Users are then free to define their own tensor transformations with the following methods:

:``transform_before_prediction``: Apply a tensor transformation to the input data before *Network* forward pass.

:``transform_before_loss``: Apply a tensor transformation to the ground truth data and the *Network* output before the
                            loss computation.

:``transform_before_apply``: Apply a tensor transformation to the *Network* output before sending the prediction to the
                             *Environment*.


Configurations
--------------

Using a *Network*, an *Optimizer* and a *DataTransformation* in one of the DeepPhysX *Pipeline* always requires a
*NetworkConfig*.
This *Configuration* object role is both to bring together all the options for configuring this set of objects and to
create them.
Objects are created within the ``create_network``, ``create_optimization`` and ``create_data_transformation`` methods
respectively.

| **Network parameters**
| Here is a description of attributes related to *Network* configuration:

:``network_class``: *Network* class from which an instance will be created.

:``network_dir``: Path to an existing *Network* repository with saved parameters.

:``network_name``: Name of the *Network* object.

:``network_type``: Type of the *Network*.

:``which_network``: If several sets of parameters are saved in ``network_dir``, load the specified one.
                    By default, the last save is loaded.

:``save_each_epoch``: If True, *Network* parameters will be saved at the end of each training epoch.
                      Otherwise, they are only saved at the end of the training session.

| **Optimization parameters**
| Here is a description of attributes related to *Optimization* configuration:

:``optimization_class``: - *Optimization* class from which an instance will be created.

:``lr``: Learning rate value.

:``loss``: Loss class that will be used to compute loss value.

:``optimizer``: *Optimizer* class that will be used to optimize the *Network* parameters.

:``require_training_stuff``: In the case where a loss class and / or an optimizer class (training stuff) are not
                             used for training, users must set this option to False.

| **DataTransformation parameters**
| Here is a description of attributes related to *DataTransformation* configuration:

:``data_transformation_class``: *DataTransformation* class from which an instance will be created.
