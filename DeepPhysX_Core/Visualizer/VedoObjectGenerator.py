from DeepPhysX_Core.Visualizer.VedoObjectFactories.VedoObjectFactory import *
from typing import Callable

create_filter: Dict[str, Callable[[ObjectDescription], VisualInstance]] = {
    "Mesh": lambda sorted_data: Mesh(inputobj=[sorted_data["positions"], sorted_data["cells"]],
                                     c=sorted_data["c"],
                                     alpha=sorted_data["alpha"],
                                     computeNormals=sorted_data["computeNormals"]),
    "Points": lambda sorted_data: Points(inputobj=sorted_data["positions"],
                                         c=sorted_data["c"],
                                         alpha=sorted_data["alpha"],
                                         r=sorted_data["r"]),
    "Marker": lambda sorted_data: Marker(symbol=sorted_data["symbol"],
                                         pos=sorted_data["position"][0],
                                         c=sorted_data["c"],
                                         alpha=sorted_data["alpha"],
                                         s=sorted_data["s"],
                                         filled=sorted_data["filled"]),
    "Glyph": lambda sorted_data: Glyph(mesh=sorted_data["positions"],
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
    """
    Container class that contain a scene description (factory, hierarchy, instances)
        Description:
            VedoObjects is container that matches factories and VisualInstances to provide an easy and intuitive mean
            to update Vedo scenes from a remote client
    """
    def __init__(self):
        """
        Automatically set the attributes of a Vedo scene
        """
        self.name = self.__class__.__name__
        self.objects_instance = {}
        self.objects_factory = VedoObjectFactory()

    def create_object(self, data_dict: ObjectDescription) -> None:
        """
        Initialise a factory and a VisualInstance with the given parameters (Type, positions, etc.)

        :param data_dict: Dict[str, Union[Dict[str, Any], Any]] Dictionary that describes the parameters and type of and object
        """
        sorted_data_dict, object_id, _ = self.objects_factory.add_object(object_type=data_dict['type'], data_dict=data_dict)
        self.objects_instance[object_id] = {"instance": create_filter[data_dict['type']](sorted_data_dict)}

    def update_objects(self, object_id: int, new_dict: ObjectDescription) -> None:
        """
        Update the factory designed by the object_id with the given data

        :param object_id: int ID of the factory/object to update
        :param new_dict: Dict[str, Dict[str, Any]] Dictionary containing the data to update
        """
        self.objects_factory.update_object_dict(object_id, new_dict)

    def update_instance(self, object_id: int) -> VisualInstance:
        """
        Update the VisualInstance designed by the object_id with the corresponding factory

        :param object_id: int ID of the factory/object to update
        :return: The update VisualInstance
        """
        return self.objects_factory.update_object_instance(object_id=object_id, instance=self.objects_instance[object_id]['instance'])

    def update(self, object_id: int, new_dict: ObjectDescription) -> VisualInstance:
        """
        Update both the factory and VisualInstance designed by the object id

        :param object_id: int ID of the factory/object to update
        :param new_dict: Dict[str, Dict[str, Any]] Dictionary containing the data to update
        :return: The update VisualInstance
        """
        self.update_objects(object_id=object_id, new_dict=new_dict)
        return self.update_instance(object_id=object_id)
