from vedo import Plotter
from os.path import join as osPathJoin
from os import makedirs

from DeepPhysX_Core.Visualizer.VedoObjectGenerator import *


class VedoVisualizer:
    """
    Visualizer class to display VisualInstances in a 2D/3D environment
        Description:
            VedoVisualiser use the vedo library to display 3D models. Objects are given in the init_view function.\n
            Updates to these objects are achieved by using updateVisualizer and updateInstances functions.
    """

    scene = Dict[int, VedoObjects]
    default_viewer_id: int
    viewers: Dict[int, Dict[str, Union[str, List[Any], bool, Plotter]]]
    folder_path: str
    nb_saved: int
    nb_screenshots: int
    
    def __init__(self):
        """
        Automatically set the default parameters of the VedoVisualiser
        """
        super().__init__()
        self.scene = {}
        self.default_viewer_id = 9
        self.viewers = {self.default_viewer_id: {'title': f"Vedo_axes_{self.default_viewer_id}", 'instances': [], 'sharecam': False, 'interactive': True}}
        # Wrong samples parameters
        self.folder_path = ""
        self.nb_saved = 0
        self.nb_screenshots = 0

    def init_view(self, data_dict: Dict[int, Dict[int, Dict[str, Union[Dict[str, Any], Any]]]]) -> None:
        """
        Initialize VedoVisualizer class by parsing the scenes hierarchy and creating VisualInstances.
        OBJECT DESCRIPTION DICTIONARY is usually obtained using the corresponding factory (VedoObjectFactory)
            data_dict example:

                {SCENE_1_ID: {OBJECT_1.1_ID: {CONTENT OF OBJECT_1.1 DESCRIPTION DICTIONARY},\n
                             ...\n
                             OBJECT_1.N_ID:  {CONTENT OF OBJECT_1.N DESCRIPTION DICTIONARY}\n
                             },\n
                ...\n
                SCENE_M_ID: OBJECT_M.1_ID: {CONTENT OF OBJECT_K.1 DESCRIPTION DICTIONARY},\n
                            ...\n
                            OBJECT_M.K_ID:  {CONTENT OF OBJECT_K.P DESCRIPTION DICTIONARY}\n
                            }\n
                }

        :param data_dict: Dict[int, Dict[int, Dict[str, Union[Dict[str, Any], Any]]]] Dictionary describing the scene hierarchy and object parameters.
        """
        from itertools import filterfalse
        for scene_id in sorted(data_dict):  # For each scene/client
            self.scene.update({scene_id: VedoObjects()})

            scene = self.scene[scene_id]

            for object_id in data_dict[scene_id]:  # For each object in current scene/client
                scene.create_object(data_dict[scene_id][object_id])

            # Simple syntax shortcuts
            objects_dict = scene.objects_factory.objects_dict

            # Deal with all of the windows and object attached to these windows first
            remaining_object_id = set(objects_dict.keys())
            for window_id in scene.objects_factory.windows_id:
                # Removes the window we are dealing with from the object we have to add to the plotter
                remaining_object_id -= {window_id}
                # Vedo can only handle 1 axe type per viewer so we create as much viewers as needed
                viewer_id = objects_dict[window_id]['axes']
                if viewer_id in self.viewers:
                    # If atleast one need a sharedcam then we wet true for all
                    self.viewers[viewer_id]['sharecam'] |= objects_dict[window_id]['sharecam']
                    # If one requires that the window is not interactive then it's not interactive.
                    self.viewers[viewer_id]['interactive'] &= objects_dict[window_id]['interactive']
                else:
                    self.viewers[viewer_id] = {}
                    self.viewers[viewer_id]['sharecam'] = objects_dict[window_id]['sharecam']
                    self.viewers[viewer_id]['interactive'] = objects_dict[window_id]['interactive']
                    self.viewers[viewer_id]['instances'] = []
                    self.viewers[viewer_id]['title'] = f"Vedo_axes_{objects_dict[window_id]['axes']}"

                # Add the objects to the corresponding list
                for object_id in objects_dict[window_id]['objects_id']:
                    # Affects the object in the existing window
                    if -1 < objects_dict[object_id]['at'] < len(self.viewers[viewer_id]['instances']):
                        self.viewers[viewer_id]['instances'][objects_dict[object_id]['at']].append([scene_id, object_id])
                    else:
                        # Affects the object in the next non existing window.
                        objects_dict[object_id]['at'] = len(self.viewers[viewer_id]['instances'])
                        self.viewers[viewer_id]['instances'].append([[scene_id, object_id]])

                # Removes all of the objects attached to the window from the object to deal with
                remaining_object_id -= set(objects_dict[window_id]['objects_id'])

            # Deals with the remaining objects that are not specified in windows
            for object_id in remaining_object_id:
                # Affects the object in the existing window
                if -1 < objects_dict[object_id]['at'] < len(self.viewers[self.default_viewer_id]['instances']):
                    self.viewers[self.default_viewer_id]['instances'][objects_dict[object_id]['at']].append([scene_id, object_id])
                else:
                    # Affects the object in the next non existing window.
                    objects_dict[object_id]['at'] = len(self.viewers[self.default_viewer_id]['instances'])
                    self.viewers[self.default_viewer_id]['instances'].append([[scene_id, object_id]])

        # Once all objects are created we create the plotter with the corresponding parameters
        for viewer_id in list(self.viewers.keys()):
            if len(self.viewers[viewer_id]['instances']) == 0:
                del self.viewers[viewer_id]
                continue

            self.viewers[viewer_id]['plotter'] = Plotter(N=len(self.viewers[viewer_id]['instances']),
                                                         title=self.viewers[viewer_id]['title'],
                                                         axes=viewer_id,
                                                         sharecam=self.viewers[viewer_id]['sharecam'],
                                                         interactive=self.viewers[viewer_id]['interactive'])

            # self.viewers[viewer_id]['instances'] is a list of list of instances
            # Each sublist contains all instances present in a window hence, each sublist has it own "at"
            for at, ids in enumerate(self.viewers[viewer_id]['instances']):
                for scene_id, object_in_scene_id in ids:
                    self.viewers[viewer_id]['plotter'].add(self.scene[scene_id].objects_instance[object_in_scene_id]['instance'], at=at, render=False)

            self.viewers[viewer_id]['plotter'].show(interactive=True)

        self.render()

    def render(self) -> None:
        """
        Call render on all valid plotter.
        """
        self.update_instances()
        for viewer_id in self.viewers:
            self.viewers[viewer_id]['plotter'].render()
            self.viewers[viewer_id]['plotter'].allowInteraction()

    def update_instances(self) -> None:
        """
        Call update_instance on all updates object description
        """
        for scene_id in self.scene:
            for object_id in self.scene[scene_id].objects_instance:
                self.scene[scene_id].update_instance(object_id)

    def update_visualizer(self, data_dict: Dict[int, Dict[int, ObjectDescription]]) -> None:
        """
        Call update_object_dict on all designed objects

        :param data_dict:
        """
        for scene_id in data_dict:
            for object_id in data_dict[scene_id]:
                self.scene[scene_id].objects_factory.update_object_dict(object_id, data_dict[scene_id][object_id])

    def save_sample(self, session_dir: str, viewer_id: int) -> None:
        """
        Save the samples as a .npz file

        :param session_dir: Directory in which to save the file
        :param viewer_id: id of the designed viewer
        """
        if self.folder_path == "":
            self.folder_path = osPathJoin(session_dir, 'dataset', 'wrong_samples')
            makedirs(self.folder_path)
            from DeepPhysX_Core.Utils import wrong_samples
            import shutil
            shutil.copy(wrong_samples.__file__, self.folder_path)
        filename = osPathJoin(self.folder_path, f'wrong_sample_{self.nb_saved}.npz')
        self.nb_saved += 1
        self.viewers[viewer_id]['plotter'].export(filename=filename)

    def save_screenshot(self, session_dir: str) -> None:
        """
        Save a screenshot of each viewer in the dataset folder_path of the session.

        :param session_dir: Directory in which to save the file
        """

        # Check folder_path existence
        if self.folder_path == "":
            self.folder_path = osPathJoin(session_dir, 'dataset', 'samples')
            makedirs(self.folder_path)

        # Save a screenshot for each viewer
        for viewer_id in self.viewers.keys():
            filename = osPathJoin(self.folder_path, f'screenshot_{self.nb_screenshots}.png')
            self.nb_screenshots += 1
            self.viewers[viewer_id]['plotter'].screenshot(filename=filename)
