CORE - Using a Visualizer
=========================

How to use
----------

DeepPhysX provides a visualization tool written with Vedo (a Python library based on Numpy and VTK) called
*VedoVisualizer*.
This *Visualizer* brings several advantages:

* Users can add any component of the simulation in the *Visualizer*;
* This *Visualizer* is compatible with every DeepPhysX *Pipeline*;
* Parallel running *Environments* are rendered in the same window with sub-windows;
* A *Factory* is created with each *Environment* so that users can access templates to define visualization data.

Objects are created using the ``add_object`` method of the *Factory* in the *Environment*.
This method requires the object type name and the data dictionary containing the required fields detailed above.
These objects must be defined in the send_visualization method of the *Environment*, which must return the objects
dictionary of the *Factory*.

Objects are updated using the ``update_object_dict`` method of the *Factory* in the *Environment*.
This method required the object index (the indices follow the order of creation) and the data dictionary containing
the updated data fields.
The *Factory* will only use templates to create an updated objects dictionary which must be sent with the request
``update_visualization`` to update the view.

| **Visual Objects**
| A list of templates are available in the *Factory* to initialize and update a list of objects.
  Here is a description of available objects and the required data fields:

* `Mesh <https://vedo.embl.es/autodocs/content/vedo/mesh.html#id1>`_ - Create a Mesh.

  +--------------------+--------------+--------------+---------------------------------------------------------+
  | **Field**          | **Init**     | **Update**   | **Description**                                         |
  +====================+==============+==============+=========================================================+
  | ``positions``      | **Required** | **Required** | List of vertices.                                       |
  |                    |              |              | Updated position vector must always have the same size. |
  +--------------------+--------------+--------------+---------------------------------------------------------+
  | ``cells``          | **Required** | Unnecessary  | List of connections between vertices.                   |
  +--------------------+--------------+--------------+---------------------------------------------------------+
  | ``computeNormals`` | Optional     | Unnecessary  | Compute cells and points normals at creation.           |
  +--------------------+--------------+--------------+---------------------------------------------------------+

* `Points <https://vedo.embl.es/autodocs/content/vedo/pointcloud.html#points>`_ - Create a Point Cloud.

  +---------------+--------------+--------------+---------------------------------------------------------+
  | **Field**     | **Init**     | **Update**   | **Description**                                         |
  +===============+==============+==============+=========================================================+
  | ``positions`` | **Required** | **Required** | List of vertices.                                       |
  |               |              |              | Updated position vector must always have the same size. |
  +---------------+--------------+--------------+---------------------------------------------------------+
  | ``r``         | Optional     | Optional     | Radius of points.                                       |
  +---------------+--------------+--------------+---------------------------------------------------------+

* `Marker <https://vedo.embl.es/autodocs/content/vedo/shapes.html#marker>`_ - Create a single Point with an associated
  symbol.

  +---------------+--------------+--------------+--------------------------------------+
  | **Field**     | **Init**     | **Update**   | **Description**                      |
  +===============+==============+==============+======================================+
  | ``positions`` | **Required** | **Required** | Position of the Marker.              |
  +---------------+--------------+--------------+--------------------------------------+
  | ``symbol``    | **Required** | Unnecessary  | Associated symbol.                   |
  +---------------+--------------+--------------+--------------------------------------+
  | ``s``         | Optional     | Unnecessary  | Radius of symbol.                    |
  +---------------+--------------+--------------+--------------------------------------+
  | ``filled``    | Optional     | Unnecessary  | Fill the shape or only draw outline. |
  +---------------+--------------+--------------+--------------------------------------+

* `Glyph <https://vedo.embl.es/autodocs/content/vedo/shapes.html#glyph>`_ - Create a Point Cloud with oriented Markers.

  +-----------------------------+--------------+--------------+------------------------------------------+
  | **Field**                   | **Init**     | **Update**   | **Description**                          |
  +=============================+==============+==============+==========================================+
  | ``positions``               | **Required** | **Required** | Position of the Markers.                 |
  +-----------------------------+--------------+--------------+------------------------------------------+
  | ``glyphObj``                | **Required** | Unnecessary  | Marker object.                           |
  +-----------------------------+--------------+--------------+------------------------------------------+
  | ``orientationArray``        | **Required** | Unnecessary  | List of orientation vectors.             |
  +-----------------------------+--------------+--------------+------------------------------------------+
  | ``scaleByScalar``           | Optional     | Unnecessary  | Glyph is scaled by the scalar field.     |
  +-----------------------------+--------------+--------------+------------------------------------------+
  | ``scaleByVectorSize``       | Optional     | Unnecessary  | Glyph is scaled by the size of the       |
  |                             |              |              | orientation vectors.                     |
  +-----------------------------+--------------+--------------+------------------------------------------+
  | ``scaleByVectorComponents`` | Optional     | Unnecessary  | Glyph is scaled by the components of the |
  |                             |              |              | orientation vectors.                     |
  +-----------------------------+--------------+--------------+------------------------------------------+
  | ``colorByScalar``           | Optional     | Unnecessary  | Glyph is colored based on the colormap   |
  |                             |              |              | and the scalar field.                    |
  +-----------------------------+--------------+--------------+------------------------------------------+
  | ``colorByVectorSize``       | Optional     | Unnecessary  | Glyph is colored based on the size       |
  |                             |              |              | of the orientation vectors.              |
  +-----------------------------+--------------+--------------+------------------------------------------+
  | ``tol``                     | Optional     | Unnecessary  | Minimum distance between two Glyphs.     |
  +-----------------------------+--------------+--------------+------------------------------------------+

* `Arrows <https://vedo.embl.es/autodocs/content/vedo/shapes.html#arrows>`_ - Create 3D Arrows.

  +---------------+--------------+--------------+----------------------------------------+
  | **Field**     | **Init**     | **Update**   | **Description**                        |
  +===============+==============+==============+========================================+
  | ``positions`` | **Required** | **Required** | Start points of the arrows.            |
  +---------------+--------------+--------------+----------------------------------------+
  | ``vectors``   | **Required** | **Required** | Vector that must represent the arrows. |
  +---------------+--------------+--------------+----------------------------------------+
  | ``res``       | Optional     | Unnecessary  | Arrows visual resolution.              |
  +---------------+--------------+--------------+----------------------------------------+

* Window -

  +---------------+--------------+--------------+----------------------------------------+
  | **Field**     | **Init**     | **Update**   | **Description**                        |
  +===============+==============+==============+========================================+
  |               |              |              |                                        |
  +---------------+--------------+--------------+----------------------------------------+

| **General parameters**
| Visual objects share default data fields that could also be filled at init and are all optional:

+-----------------------+----------+-------------+-----------------------------------------------------+
| **Field**             | **Init** | **Update**  | **Description**                                     |
+=======================+==========+=============+=====================================================+
| ``c``                 | Optional | Unnecessary | Opacity of the object between 0 and 1.              |
+-----------------------+----------+-------------+-----------------------------------------------------+
| ``alpha``             | Optional | Unnecessary | Marker object.                                      |
+-----------------------+----------+-------------+-----------------------------------------------------+
| ``at``                | Optional | Unnecessary | Sub-window in which the object will be rendered.    |
|                       |          |             | Set to -1 by default, meaning a new window is       |
|                       |          |             | created for the object.                             |
|                       |          |             | In general, to set several objects from the same    |
|                       |          |             | *Environment* in a common sub-window, set the value |
|                       |          |             | to the *Environment* instance number.               |
+-----------------------+----------+-------------+-----------------------------------------------------+
| ``colormap``          | Optional | Unnecessary | Name of color palette that samples a continuous     |
|                       |          |             | function between two end colors.                    |
+-----------------------+----------+-------------+-----------------------------------------------------+
| ``scalar_field``      | Optional | Unnecessary | List of scalar values to set individual points or   |
|                       |          |             | cell color.                                         |
+-----------------------+----------+-------------+-----------------------------------------------------+
| ``scalar_field_name`` | Optional | Unnecessary | Name of the scalar field.                           |
+-----------------------+----------+-------------+-----------------------------------------------------+


Configuration
-------------

Configuring a *Visualizer* is very simple, since the only option to change is the ``visualizer`` field in the
*EnvironmentConfig*.
If set to None, no *Visualizer* will be created, even if the *Environment* uses its *Factory* to create and update
visualization data.
It must be set to *VedoVisualizer* to activate the visualization tool.
