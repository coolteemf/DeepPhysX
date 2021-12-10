from DeepPhysX_Core.Visualizer.VedoObjectFactories.MarkerFactory import MarkerFactory
from DeepPhysX_Core.Visualizer.VedoObjectFactories.MeshFactory import MeshFactory
from DeepPhysX_Core.Visualizer.VedoObjectFactories.GlyphFactory import GlyphFactory
from DeepPhysX_Core.Visualizer.VedoObjectFactories.PointsFactory import PointsFactory
from DeepPhysX_Core.Visualizer.VedoObjectFactories.WindowFactory import WindowFactory


def factory_getter(object_type: str):
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


class VedoObjectFactory:
    def __init__(self):
        self.next_ID = 0
        self.objects_dict = {}
        self.windows_id = []
        self.factories = {}

    def addObject(self, object_type: str, data_dict: dict):
        self.factories[self.next_ID] = factory_getter(object_type)
        if self.factories[self.next_ID] is None:
            raise ValueError("The given type does not exist. Please use on of the following :\n"
                             "Mesh, mesh\n"
                             "Points, PointCloud, Point, points, point\n"
                             "Glyph, glyph\n"
                             "Marker, marker, markers, Markers\n"
                             "Window, window\n")
        self.objects_dict[self.next_ID] = self.factories[self.next_ID].parse(data_dict)
        if object_type in ['Window', 'window']:
            self.windows_id.append(self.next_ID)
        self.next_ID += 1
        return self.objects_dict[self.next_ID - 1], self.next_ID - 1, self.factories[self.next_ID - 1]

    def updateObject_dict(self, object_id: int, new_data_dict: dict):
        if object_id not in self.factories:
            self.factories[object_id] = factory_getter(self.objects_dict[object_id]["type"])
        self.objects_dict[object_id] = self.factories[object_id].update_dict(new_data_dict)
        return self.objects_dict[object_id], self.factories[object_id]

    def updateObject_instance(self, object_id: int, instance):
        if object_id not in self.factories:
            self.factories[object_id] = factory_getter(self.objects_dict[object_id]["type"])
        return self.factories[object_id].update_instance(instance)