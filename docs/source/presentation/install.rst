Install
=======

.. role:: core
.. role:: simu
.. role:: ai

It is recommended to read the :ref:`overview-packages` section first to understand the packages organisation.

The :core:`CORE` package :guilabel:`DeepPhysX_Core` is always installed by default.

Using a :simu:`SIMULATION` package is recommended but not required.

At least one :ai:`AI` package is mandatory (default one is :guilabel:`DeepPhysX_Torch`).


Prerequisites
-------------

**DeepPhysX** packages have the following dependencies :

.. table::
    :widths: 20 20 10 30

    +-----------------------------+-------------------------------+--------------+-------------------------------------+
    | **Package**                 | **Dependency**                | **Type**     | **Install**                         |
    +=============================+===============================+==============+=====================================+
    | :guilabel:`DeepPhysX_Core`  | :Numpy:`Numpy <>`             | **Required** | ``pip install numpy``               |
    |                             +-------------------------------+--------------+-------------------------------------+
    |                             | :Vedo:`Vedo <>`               | **Required** | ``pip install vedo``                |
    |                             +-------------------------------+--------------+-------------------------------------+
    |                             | :Tensorboard:`Tensorboard <>` | **Required** | ``pip install tensorboard``         |
    +-----------------------------+-------------------------------+--------------+-------------------------------------+
    | :guilabel:`DeepPhysX_Sofa`  | :SOFA:`SOFA Framework <>`     | **Required** | :SOFAI:`Follow instructions <>`     |
    |                             +-------------------------------+--------------+-------------------------------------+
    |                             | :SP3:`SofaPython3 <>`         | **Required** | :SP3I:`Follow instructions <>`      |
    |                             +-------------------------------+--------------+-------------------------------------+
    |                             | :Caribou:`Caribou <>`         | Optional     | :CaribouI:`Follow instructions <>`  |
    +-----------------------------+-------------------------------+--------------+-------------------------------------+
    | :guilabel:`DeepPhysX_Torch` | :PyTorch:`PyTorch <>`         | **Required** | ``pip install torch``               |
    +-----------------------------+-------------------------------+--------------+-------------------------------------+

.. note::
    :guilabel:`DeepPhysX_Sofa` has a dependency to :Caribou:`Caribou <>` to run the demo scripts from
    ``Examples/SOFA/Demo`` since implemented simulations involve some of its components.

Install
-------

Start by cloning the **DeepPhysX** source code from its Github repository:

.. code-block:: bash

    $ git clone https://github.com/mimesis/deepphysx.git
    $ cd DeepPhysX

Specify which packages to install by running the setup script with ``<package_name>=<do_install>`` variables:

.. code-block:: bash

    # Example 0: Default configuration
    $ python3 config.py
    >> Applied configuration with values:
            PACKAGE_CORE:  True
            PACKAGE_SOFA:  False
            PACKAGE_TORCH: True

    # Example 1: Installing only Torch package
    $ python3 config.py torch=1 sofa=0
    >> Applied configuration with values:
            PACKAGE_CORE:  True
            PACKAGE_SOFA:  False    # Might be True if already installed
            PACKAGE_TORCH: True

    # Example 2: Installing only Sofa package (another AI package must be installed)
    $ python3 config.py torch=False sofa=True
    >> Applied configuration with values:
            PACKAGE_CORE:  True
            PACKAGE_SOFA:  True
            PACKAGE_TORCH: False    # Might be True if already installed

.. note::
    Setup script will also look for already installed **DeepPhysX** packages to keep them set as installed in the setup.

Finally install the defined packages using ``pip``:

.. code-block:: bash

    pip3 install .

    # or in editor mode:
    pip3 install -e .


