DeepPhysX
=========

**Interfacing AI with simulation**

The **DeepPhysX** project provides Python modules allowing users to easily interface their **numerical simulations**
with **learning algorithms**.

DeepPhysX is mainly designed for `SOFA <https://www.sofa-framework.org/>`_ and `PyTorch <https://pytorch.org/>`_
frameworks, but other simulation and AI frameworks can be used.
For more information about DeepPhysX design, reading the section :doc:`presentation/overview` is **highly recommended**.

.. toctree::
    :caption: PRESENTATION
    :maxdepth: 2
    :hidden:

    About     <presentation/about.rst>
    Overview  <presentation/overview.rst>
    Install   <presentation/install.rst>

.. toctree::
    :caption: CORE
    :maxdepth: 2
    :hidden:

    Pipelines    <core/pipelines.rst>
    Environment  <core/environment.rst>
    Network      <core/network.rst>
    Dataset      <core/dataset.rst>
    Visualizer   <core/visualizer.rst>
    Stats        <core/stats.rst>

.. toctree::
    :caption: API
    :maxdepth: 1
    :hidden:

    Core  <api/core.rst>
    Sofa  <api/sofa.rst>
