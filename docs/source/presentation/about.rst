About
=====

The purpose of the **DeepPhysX framework** is to provide an interface between **Deep Learning algorithms** and
**numerical simulations**.

This full Python3 project brings several pipelines, allowing the user to:

* Generate synthetic data from simulations
* Train Artificial Neural Networks with the synthetic Dataset
* Use the predictions of trained Artificial Neural Networks inside the simulations.

.. note::
    The Dataset generation and the training pipelines can be done simultaneously.

DeepPhysX manages not only the production of synthetic data, which can be achieved within several simulations running
in multiprocessing, but also the storage of the produced Dataset.

Additional visualization tools are provided during the training sessions:

* A visualization Factory to gather the rendering of all the simulated objects (written with `Vedo
  <https://vedo.embl.es/>`_)
* An analysis of the evolution of the training session (written with `Tensorboard
  <https://www.tensorflow.org/tensorboard>`_)
