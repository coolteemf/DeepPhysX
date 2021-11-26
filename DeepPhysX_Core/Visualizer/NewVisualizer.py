from typing import Dict, List, Any, Union
from vedo import Plotter
from os.path import join as osPathJoin
from os import makedirs

from DeepPhysX_Core.Visualizer.VedoObjectGenerator import VedoObjects


class NewVisualizer:
    viewers: Dict[int, Dict[str, Union[str, List[Any], bool, Plotter]]]

    def __init__(self):
        """
        Display added objects. Mostly used to visualize data during training or prediction phase directly from the network.
        :param str title: Name of the window
        """
        super().__init__()
        self.scene = {}  # objects[CLIENT_ID] -> VedoObjects (objects_instance[OBJECT_IN_CLIENT_ID] -> instance , objects_factory -> objects_dict[OBJECT_IN_CLIENT_ID] -> {})
        self.viewers = {2: {'title': "Vedo_axes_2", 'instances': [], 'sharecam': False, 'interactive': True}}
        # Wrong samples parameters
        self.folder = None
        self.nb_saved = 0

    def initView(self, data_dict):
        """
        Add all objects described in data_dict.

        :param data_dict: Dictionary containing the meshes data for each environment.
        :return:
        """
        from itertools import filterfalse
        for scene_id in sorted(data_dict):  # For each scene/client
            self.scene.update({scene_id: VedoObjects()})

            scene = self.scene[scene_id]

            for object_id in data_dict[scene_id]:  # For each object in current scene/client
                scene.createObject(data_dict[scene_id][object_id])

            # Simple syntax shortcuts
            objects_dict = scene.objects_factory.objects_dict

            # Deal with all of the windows and object attached to these windows first
            remaining_object_id = list(objects_dict.keys())
            for window_id in scene.objects_factory.windows_id:
                print(f"Removing window id {window_id}")
                remaining_object_id.remove(window_id)
                # Vedo can only handle 1 axe type per viewer so we create as much viewers as needed
                viewer_id = objects_dict[window_id]['axes']
                viewer = self.viewers[viewer_id]
                if viewer_id in self.viewers:
                    print("Viewer already exist we simply add the object parameters")
                    # If atleast one need a sharedcam then we wet true for all
                    viewer['sharecam'] |= objects_dict[window_id]['sharecam']
                    # If one requires that the window is not interactive then it's not interactive.
                    viewer['interactive'] &= objects_dict[window_id]['interactive']
                else:
                    print("Viewer do not exist we set parameters")
                    viewer['sharecam'] = objects_dict[window_id]['sharecam']
                    viewer['interactive'] = objects_dict[window_id]['interactive']
                    viewer['instances'] = []
                    viewer['title'] = f"Vedo_axes_{objects_dict[window_id]['axes']}"

                # CA VA SANS DOUTE BUGGER DANS LE COIN
                remaining_object_id = list(
                    filterfalse(lambda x: x not in objects_dict[window_id]['objects_id'], remaining_object_id))
                viewer['instances'].append([scene_id, objects_dict[window_id]['objects_id']])
                # JUSQU'A LA

            # Deals with the remaining objects that are not specified in windows
            for object_id in remaining_object_id:
                # Affects the object in the existing window
                if -1 < objects_dict[object_id]['at'] < len(self.viewers[2]['instances']):
                    self.viewers[2]['instances'][objects_dict[object_id]['at']].append([scene_id, object_id])
                else:
                    # Affects the object in the next non existing window.
                    objects_dict[object_id]['at'] = len(self.viewers[2]['instances'])
                    self.viewers[2]['instances'].append([[scene_id, object_id]])

        # Once all objects are created we create the plotter with the corresponding parameters
        for viewer_id in self.viewers:
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

    def render(self):
        """
        Render the meshes in the desired windows.
        :return:
        """
        self.updateInstances()
        for viewer_id in self.viewers:
            self.viewers[viewer_id]['plotter'].render()
            self.viewers[viewer_id]['plotter'].allowInteraction()

    def updateInstances(self):
        for scene_id in self.scene:
            for object_id in self.scene[scene_id].objects_instance:
                self.scene[scene_id].updateInstance(object_id)

    def updateVisualizer(self, data_dict: dict):
        for scene_id in data_dict:
            for object_id in data_dict[scene_id]:
                self.scene[scene_id].objects_factory.updateObject_dict(object_id, data_dict[scene_id][object_id])

    def saveSample(self, session_dir, viewer_id):
        """
        Save the samples as a .npz file

        :param str session_dir: Directory in which to save the file

        :return:
        """
        if self.folder is None:
            self.folder = osPathJoin(session_dir, 'stats/wrong_samples')
            makedirs(self.folder)
            from DeepPhysX_Core.utils import wrong_samples
            import shutil
            shutil.copy(wrong_samples.__file__, self.folder)
        filename = osPathJoin(self.folder, f'wrong_sample_{self.nb_saved}.npz')
        self.nb_saved += 1
        self.viewers[viewer_id]['plotter'].export(filename=filename)
