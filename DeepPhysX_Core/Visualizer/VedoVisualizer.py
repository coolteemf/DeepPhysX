import numpy as np
import vedo
import os

from DeepPhysX_Core.Visualizer.VedoObject import VedoObject


class VedoVisualizer:

    def __init__(self, visual_object=VedoObject, title='VedoVisualizer', interactive_window=False, show_axes=False,
                 min_color='yellow', max_color='red', range_color=10):
        """
        Base class to display Vedo objects. Used to visualize data during training or prediction phase.

        :param title: Name of the window
        :param interactive_window: If True the Visualizer will be interactive
        :param show_axes: If Ture display axes
        :param min_color: Color of the min values of a mesh. Can be described as a list of 3 or 4 float using
        RGB and RGBA convention
        :param max_color: Color of the max values of a mesh. Can be described as a list of 3 or 4 float using
        RGB and RGBA convention
        :param range_color: Number of interpolations between min and max color
        """
        self.name = self.__class__.__name__
        # Models data
        self.objects = {}
        self.vedo_object = visual_object(visualizer=self)
        self.scenes = []
        self.shapes = []
        # Viewer data
        self.viewer = None
        self.nb_view = 0
        self.colormap = vedo.buildPalette(color1=min_color, color2=max_color, N=range_color, hsv=False)
        self.window_parameters = {'title': title, 'interactive': interactive_window, 'axes': show_axes}
        # Wrong sample saving data
        self.folder = None
        self.nb_saved = 0

    def init(self, data_dict):
        """
        Add all objects described in data_dict.

        :param data_dict: Dictionary containing the whole visualization data.
        :return:
        """
        # List containing all the objects created for each environment
        all_objects = []

        # Loop on the environments
        for idx in data_dict:
            # A scene is the set of objects in an environment
            scene = data_dict[idx]
            objects, scene_shapes = self.vedo_object.createObjects(scene)
            self.scenes.append(objects)
            self.shapes.append(scene_shapes)
            all_objects += objects

        self.viewer = vedo.Plotter(N=self.nb_view, title=self.window_parameters['title'],
                                   axes=self.window_parameters['axes'], sharecam=True,
                                   interactive=self.window_parameters['interactive'])
        for obj in all_objects:
            self.viewer.add(obj, at=self.objects[obj]['at'])
        print(f"[{self.name}] Set visualizer camera position then close visualizer window to start.")
        self.viewer.show(interactive=True)
        self.render()

    def render(self):
        """
        Render the meshes in the desired windows.

        :return:
        """
        self.update()
        self.viewer.render()
        self.viewer.allowInteraction()

    def update(self, position=True, scalar_field=True, cells=False):
        """

        :param position:
        :param scalar_field:
        :param cells:
        :return:
        """
        self.vedo_object.updateObjects()

    def updateFromBatch(self, batch):
        """
        update vedo environment using batch information

        :param batch: dict templated as
                        {0: {client_parameters},
                         1: {client_parameters},
                         ...
                         N-1: {client_parameters},
                         'in': input_learning data,
                         'out': output_learning data}

                      client_parameter contain all data sent by the environment to the environment manager (position, velocity, strain, etc...)
        :return:
        """
        # assume client order == vedo mesh order
        for idx, sample in enumerate(batch):
            self.updateFromSample(sample, idx)

    def updateFromSample(self, sample, idx):
        """
        Update vedo meshes using a sample from an environment.

        :param sample: Dict templated as {'position': [position_mesh_1, ..., position_mesh_N]}
        :param idx: Index of environment
        :return:
        """
        # Get the list of meshes recorded in vedo for the environment n°idx
        scene = self.scenes[idx]
        # Loop on sample.keys()
        for key in sample:
            # Check that the key is used to update a mesh
            if key not in self.shapes[idx]:
                raise ValueError(f"[{self.name}] Field {key} is missing to update from sample.")
            # Reshape the array according to the number of meshes and the field's vector size
            sample[key] = sample[key].reshape(self.shapes[idx][key])
            # Check that the field contains an array for each object
            if len(sample[key]) != len(scene):
                raise ValueError(f"[{self.name}] The number of value in received data does not match the number of"
                                 f"objects for field {key}.")
            # Update the field for each object
            for i in range(len(scene)):
                self.objects[scene[i]][key] = sample[key][i]
        # Update view
        self.render()

    def saveSample(self, session_dir):
        """
        Save the samples as a .npz file

        :param str session_dir: Directory in which to save the file

        :return:
        """
        if self.folder is None:
            self.folder = os.path.join(session_dir, 'stats/wrong_samples')
            os.makedirs(self.folder)
            from DeepPhysX_Core.utils import wrong_samples
            import shutil
            shutil.copy(wrong_samples.__file__, self.folder)
        filename = os.path.join(self.folder, f'wrong_sample_{self.nb_saved}.npz')
        self.nb_saved += 1
        self.viewer.export(filename=filename)
