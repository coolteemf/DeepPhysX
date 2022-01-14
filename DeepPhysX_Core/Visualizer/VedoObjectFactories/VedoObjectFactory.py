from DeepPhysX_Core.Visualizer.VedoObjectFactories.MarkerFactory import MarkerFactory
from DeepPhysX_Core.Visualizer.VedoObjectFactories.MeshFactory import MeshFactory
from DeepPhysX_Core.Visualizer.VedoObjectFactories.GlyphFactory import GlyphFactory
from DeepPhysX_Core.Visualizer.VedoObjectFactories.PointsFactory import PointsFactory
from DeepPhysX_Core.Visualizer.VedoObjectFactories.WindowFactory import WindowFactory
from vedo import Mesh, Glyph, Marker, Points
from typing import List, Dict, Union, Tuple, Any

ObjectDescription = Dict[str, Union[Any, Dict[str, Any]]]
VisualInstance = Union[Mesh, Glyph, Marker, Points]
Factory = Union[MeshFactory, PointsFactory, GlyphFactory, MarkerFactory, WindowFactory]


class VedoObjectFactory:
    """
    Class that contains all the Factories of a scene
        Description:
            VedoObjectFactory Contains all the factories present in a Vedo scene.\n
    """
    next_id: int
    object_dict: Dict[int, ObjectDescription]
    updated_object_dict: Dict[int, ObjectDescription]
    windows_id = List[int]
    factories = Dict[int, Factory]

    def __init__(self):
        """
        Automatically set the attributes of a Vedo scene
        """
        self.next_id = 0
        self.objects_dict = {}
        self.updated_object_dict = {}
        self.windows_id = []
        self.factories = {}

    def add_object(self, object_type: str, data_dict: ObjectDescription) -> Tuple[ObjectDescription, int, Factory]:
        """
        Create a factory with the given object type and data dictionary

        :param object_type: str Type the desired factory Object (Mesh, Points, Glyph, Marker, Window)
        :param data_dict: Dict[str, Union[Dict[str, Any], Any]] Dictionary that contains the associated data
        :return: Tuple[ObjectDescription, int, Factory] The fully parsed and updated dictionary, its Id, the associated factory
        """

        self.factories[self.next_id] = self.factory_getter(object_type)

        if self.factories[self.next_id] is None:
            raise ValueError("The given type does not exist. Please use on of the following :\n"
                             "Mesh, mesh\n"
                             "Points, PointCloud, Point, points, point\n"
                             "Glyph, glyph\n"
                             "Marker, marker, markers, Markers\n"
                             "Window, window\n")

        self.objects_dict[self.next_id] = self.factories[self.next_id].parse(data_dict)

        if object_type in ['Window', 'window']:
            self.windows_id.append(self.next_id)

        self.next_id += 1

        return self.objects_dict[self.next_id - 1], self.next_id - 1, self.factories[self.next_id - 1]

    def update_object_dict(self, object_id: int, new_data_dict: ObjectDescription) -> Tuple[ObjectDescription, Factory]:
        """
        Update the object with the given ID using the data passed by new_data_dict

        :param object_id: int ID of the object to update
        :param new_data_dict: Dict[str, Union[Dict[str, Any], Any]] Dictionary containing the data to update

        :return: Tuple[ObjectDescription, Factory] The updated dictionary, the updated factory
        """
        if object_id not in self.factories:
            self.factories[object_id] = self.factory_getter(self.objects_dict[object_id]["type"])
        self.objects_dict[object_id] = self.factories[object_id].update_dict(new_data_dict)
        self.updated_object_dict[object_id] = {}
        for field in new_data_dict:
            self.updated_object_dict[object_id][field] = new_data_dict[field]
        return self.objects_dict[object_id], self.factories[object_id]

    def update_object_instance(self, object_id: int, instance: VisualInstance) -> VisualInstance:
        """
        Update the given instance using the factory corresponding to the passed object_id

        :param object_id: int ID of the factory of the object to use
        :param instance: VisualInstance object to update
        :return: The updated VisualInstance
        """
        if object_id not in self.factories:
            self.factories[object_id] = self.factory_getter(self.objects_dict[object_id]["type"])

        updated_instance = self.factories[object_id].update_instance(instance)
        self.updated_object_dict[object_id] = {}
        return updated_instance

    @staticmethod
    def factory_getter(object_type: str) -> Factory:
        """
        Helper function that return a default Factory corresponding to the given object_type

        :param object_type: str Type of the object
        :return: Factory corresponding to the given object type
        """
        factory = None
        if object_type in ["Mesh", "mesh"]:
            factory = MeshFactory()
        elif object_type in ["Points", "PointCloud", "Point", "points", "point"]:
            factory = PointsFactory()
        elif object_type in ["Glyph", "glyph"]:
            factory = GlyphFactory()
        elif object_type in ['Marker', 'marker', 'markers', "Markers"]:
            factory = MarkerFactory()
        elif object_type in ['Window', 'window']:
            factory = WindowFactory()
        return factory
