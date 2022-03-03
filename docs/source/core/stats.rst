CORE - Using the Stats
======================

How to Use
----------

During the training session, some optimizations analysis are performed and stored as a readable file for Tensorboard.
This file is automatically filled at each network optimization step in the ``stats`` repository of the current working
directory.

Tensorboard will be automatically launched for the current training session in a web browser.
To open a previous training session analysis, Tensorboard can be launched using ``tensorboard --logdir .`` in the
repository containing the stats data.

Adding fields
-------------

By default, tree data fields will be stored:

* The value of the loss function at each batch / epoch
* The mean of the loss function at each batch / epoch over the last 50 elements
* The variance of the loss function at each batch / epoch

Custom data fields can be set in Tensorboard as well.
After an optimization step, the *Optimization* class will by default return a dictionary with a single item named
``loss`` containing the current loss value.
Then, the Trainer will add every item contained in this dictionary to the *StatsManager*.
To fill this dictionary with custom fields, a custom *Optimization* class must be implemented where the
``transform_loss`` method will fill that dictionary.
This new field will have the same name as the item and take each value given in this item at each optimization step.
