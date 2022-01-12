import numpy
from vedo import utils, Mesh, Glyph, Marker, Points
import numpy as np
from typing import List, Dict, Any, Callable, TypeAlias, ParamSpec, Union

ObjectDescription: TypeAlias = Dict[str, Any]
VisualInstance: TypeAlias = Union[Mesh, Glyph, Marker, Points]
ParseParameters = ParamSpec('ParseParameters')
UpdateParameters = ParamSpec('UpdateParameters')


class parse_wrapper:
    """
    parse_wrapper is a class wrapper.

        Description:
            Wraps the parse function of all vedo BaseObjectFactory subclasses.
            Allows to only define the specialized parse procedure in the subclasses while still running the general parse procedure.
            When calling the parse function the caller is passed as args[0] (self), and the arguments are either in args or kwargs

            This class only implements __call__
    """
    def __call__(self, specialized_parse: Callable[[ParseParameters], None]) -> Callable[[ParseParameters.args, ParseParameters.kwargs], ObjectDescription]:
        """
        Function That wraps the parse call.
        Parses all the attributes shared between subclasses of BaseObjectFactory

        :param specialized_parse: Callable[[ParseParameters], None] parse function of a subclass of BaseObjectFactory

        :return: general_update: Callable[[ParseParameters.args, ParseParameters.kwargs], ObjectDescription] The general parse function
        """
        def general_parse(*args: ParseParameters.args, **kwargs: ParseParameters.kwargs) -> ObjectDescription:
            """
            Run the general parse procedure and also launch the one defined in the calling subclass of BaseObjectFactory

            :param args: Tuple[CallingSubClass,...] List with the calling subclass and maybe the dictionary of data to be parsed
            :param kwargs: Dict[str, Any] "Any" here will be the dictionary of data to be parsed if not present in args
            :return: parsed_data: Dict[str, Any] The fully parsed and evaluated dictionary
            """
            # parse only takes 2 arguments, self and data_dict
            if len(args) + len(kwargs) != 2:
                return {"": None}
            object_factory = args[0]
            if len(args) == 2:
                data_dict = args[1]
            else:
                (_, data_dict), *rest = kwargs.items()  # : (Any, ObjectDescription, Any)
            # Load all parameters directly given with the appropriate name
            object_factory.parsed_data.update(
                {word: data_dict[word] for word in object_factory.grammar if word in data_dict})
            object_factory.parsed_data["type"] = object_factory.type
            # Run the specialized parse code
            specialized_parse(self=object_factory, data_dict=data_dict)
            # Default init of all not given parameters
            for word in object_factory.grammar:
                if word not in object_factory.parsed_data:
                    object_factory.parsed_data[word] = object_factory.default_values[word]
            return object_factory.parsed_data

        return general_parse


class update_wrapper:
    """
    update_wrapper is a class wrapper.
        Description:
            Wraps the update_instance function of all vedo BaseObjectFactory subclasses.
            Allows to only define the specialized update_instance procedure in the subclasses while still running the general update_instance procedure.

            When calling the update_instance function the caller is passed as args[0] (self), and the arguments are either in args or kwargs

            This class only implements __call__
    """
    def __call__(self, specialized_update: Callable[[UpdateParameters], None]) -> Callable[[UpdateParameters.args, UpdateParameters.kwargs], VisualInstance]:
        """
        Function That wraps the update_instance call.
        Updates all of the attributes shared between subclasses of BaseObjectFactory

        :param specialized_update: Callable[[UpdateParameters], None] update_instance function of a subclass of BaseObjectFactory

        :return: general_update: Callable[[UpdateParameters.args, UpdateParameters.kwargs], Any] The general update function
        """
        def general_update(*args: UpdateParameters.args, **kwargs: UpdateParameters.kwargs) -> VisualInstance:
            if len(args) + len(kwargs) != 2:  # self, instance
                return
            # First object is always self, it is passed implicitly
            object_factory = args[0]
            # Either in kwargs or 2nd argument in args but always last with update_instance definition
            if "instance" in kwargs:
                instance = kwargs["instance"]
            else:
                instance = args[-1]
            if "colormap" in object_factory.dirty_fields:
                instance.c(object_factory.parsed_data["colormap"])
                object_factory.dirty_fields.remove("colormap")
            if "scalar_field" in object_factory.dirty_fields:
                instance.addPointArray(input_array=object_factory.parsed_data["scalar_field"],
                                       name=object_factory.parsed_data["scalar_field_name"])
                object_factory.dirty_fields.remove("scalar_field")
                object_factory.dirty_fields.remove("scalar_field_name")
            if "alpha" in object_factory.dirty_fields:
                instance.alpha(object_factory.parsed_data["alpha"])
                object_factory.dirty_fields.remove("alpha")
            if "c" in object_factory.dirty_fields:
                instance.c(object_factory.parsed_data["c"])
                object_factory.dirty_fields.remove("c")

            specialized_update(self=object_factory, instance=instance)

            if object_factory.dirty_fields:  # Check for emptyness
                print(f"Update successful ! Unused field(s) : {object_factory.dirty_fields}")

            return instance

        return general_update


class BaseObjectFactory:
    """
    Base class of all the Visual object factories.
        Description:
            BaseObjectFactory defines the parse and update procedures of all the object factories.
    """
    grammar: List[str]
    default_values: ObjectDescription
    parsed_data: ObjectDescription
    dirty_fields: List[str]
    type: str

    def __init__(self):
        """
        Automatically set all the common attributes of the factories
        """
        self.grammar = ['c', 'alpha', 'at', 'colormap', 'scalar_field', "scalar_field_name"]
        self.default_values = {self.grammar[0]: 'b', self.grammar[1]: 1.0, self.grammar[2]: -1,
                               self.grammar[3]: 'jet',
                               self.grammar[4]: [], self.grammar[5]: "scalar_field"}
        self.parsed_data = {}
        self.dirty_fields = []
        self.type = ""

    @parse_wrapper()
    def parse(self, data_dict: ObjectDescription) -> None:  # It's the wrapper that return the parsed_data
        """
        Parse the given dictionary and fill the parsed_data member accordingly

        :param data_dict: Dict[str, Any] Dictionary to parse
        :return: A Dict[str, Any] that represent the parsed_data member
        """
        raise NotImplementedError

    def update_dict(self, new_data: ObjectDescription) -> ObjectDescription:
        """
        Parse the given dictionary and update the parsed_data member accordingly

        :param new_data: Dict[str, Any] Dictionary containing the data to update
        :return: A Dict[str, Any] that represent the updated parsed_data member
        """
        self.dirty_fields = [*new_data.keys()]
        return self.parse(new_data)

    def get_data(self) -> ObjectDescription:
        """
        :return: A Dict[str, Any] that represent the parsed_data member
        """
        return self.parsed_data

    @update_wrapper()
    def update_instance(self, instance: VisualInstance) -> None:
        """
        Update the given VisualInstance instance

        :param instance: VisualInstance: Vedo object to update with its current parsed_data values
        :return: The updated VisualInstance
        """
        raise NotImplementedError

    @staticmethod
    def parse_position(data_dict: ObjectDescription, wrap: bool = True) -> numpy.ndarray:
        """
        Helper function (static method) that parses the positions field

        :param data_dict: Dict[str, Any] Dictionary to parse
        :param wrap: bool When True add a dimension to the positions data
                          ex: if the input data shape is [N, 3], and wrap is True
                              then data will have shape [1, N, 3]
        :return: A numpy.ndarray that contains the parsed position and an additional dimension if wrap is True
        """
        # Look for position.s and cell.s to make inputobj
        pos = None

        # Either positions and cells have been passed as independant array or are inexistant
        if 'position' in data_dict:
            pos = data_dict['position']
        elif 'positions' in data_dict:
            pos = data_dict['positions']

        if utils.isSequence(pos):  # passing point coords
            if not utils.isSequence(pos[0]) and wrap:
                pos = [pos]
            n = len(pos)

            if n == 3:  # assume pos is in the format [all_x, all_y, all_z]
                if utils.isSequence(pos[0]) and len(pos[0]) > 3:
                    pos = np.stack((pos[0], pos[1], pos[2]), axis=1)
            elif n == 2:  # assume pos is in the format [all_x, all_y, 0]
                if utils.isSequence(pos[0]) and len(pos[0]) > 3:
                    pos = np.stack((pos[0], pos[1], np.zeros(len(pos[0]))), axis=1)
                else:
                    pos = np.array([pos[0], pos[1], 0])

            if n and wrap and len(pos[0]) == 2:  # make it 3d
                pos = np.c_[np.array(pos), np.zeros(len(pos))]

        return pos

    def update_position(self, instance: VisualInstance) -> None:
        """
        Helper function that update the position field of the passed visual instance

        :param instance: VisualInstance Vedo object to update with its current parsed_data values
        :return: None
        """
        def test_update_function(field_name: str):
            if field_name in self.dirty_fields and field_name in self.parsed_data:
                instance.points(self.parsed_data[field_name])
                self.dirty_fields.remove(field_name)
        test_update_function('position')
        test_update_function('positions')

