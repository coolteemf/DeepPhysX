from vedo import Mesh, Points, Marker, Glyph
from DeepPhysX_Core.Visualizer.VedoObjectFactories.VedoObjectFactory import VedoObjectFactory
import numpy as np
create_filter = {
    "Mesh": lambda sorted_data: Mesh(inputobj=sorted_data["inputobj"],
                                     c=sorted_data["c"],
                                     alpha=sorted_data["alpha"],
                                     computeNormals=sorted_data["computeNormals"]),
    "Points": lambda sorted_data: Points(inputobj=sorted_data["inputobj"],
                                         c=sorted_data["c"],
                                         alpha=sorted_data["alpha"],
                                         r=sorted_data["r"]),
    "Marker": lambda sorted_data: Marker(symbol=sorted_data["symbol"],
                                         pos=sorted_data["pos"],
                                         c=sorted_data["c"],
                                         alpha=sorted_data["alpha"],
                                         s=sorted_data["s"],
                                         filled=sorted_data["filled"]),
    "Glyph": lambda sorted_data: Glyph(mesh=sorted_data["mesh"],
                                       glyphObj=create_filter['Marker'](sorted_data["glyphObj"]),
                                       orientationArray=sorted_data["orientationArray"],
                                       scaleByScalar=sorted_data["scaleByScalar"],
                                       scaleByVectorSize=sorted_data["scaleByVectorSize"],
                                       scaleByVectorComponents=sorted_data["scaleByVectorComponents"],
                                       colorByScalar=sorted_data["colorByScalar"],
                                       colorByVectorSize=sorted_data["colorByVectorSize"],
                                       tol=sorted_data["tol"],
                                       c=sorted_data["c"],
                                       alpha=sorted_data["alpha"]),
    "Window": lambda sorted_data: None  # Window is not a vedo object hence we create a pass through
}


class VedoObjects:

    def __init__(self):
        self.name = self.__class__.__name__
        self.objects_instance = {}
        self.objects_factory = VedoObjectFactory()

    def createObject(self, data_dict: dict):
        sorted_data_dict, object_id, _ = self.objects_factory.addObject(object_type=data_dict['type'], data_dict=data_dict)
        self.objects_instance[object_id] = {"instance": create_filter[data_dict['type']](sorted_data_dict)}

    def updateObjects(self, object_id: int, new_dict: dict):
        self.objects_factory.updateObject_dict(object_id, new_dict)

    def updateInstance(self, object_id: int):
        return self.objects_factory.updateObject_instance(object_id=object_id, instance=self.objects_instance[object_id]['instance'])

    def update(self, object_id: int, new_dict: dict):
        self.updateObjects(object_id=object_id, new_dict=new_dict)
        self.updateInstance(object_id=object_id)
