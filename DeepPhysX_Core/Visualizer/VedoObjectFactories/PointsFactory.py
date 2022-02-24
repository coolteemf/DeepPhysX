from DeepPhysX_Core.Visualizer.VedoObjectFactories.BaseObjectFactory import *


class PointsFactory(BaseObjectFactory):

    def __init__(self):
        """
        PointsFactory is a class that represent the data of a Points visual object.
        PointsFactory defines the parse and update procedures of Points object according to Vedo.
        """

        BaseObjectFactory.__init__(self)

        self.type = 'Points'
        self.grammar_plug = ['positions', 'r']
        self.grammar.extend(self.grammar_plug)
        self.default_values.update({self.grammar_plug[0]: None,
                                    self.grammar_plug[1]: 4})

    @parse_wrapper()
    def parse(self, data_dict: ObjectDescription) -> None:
        """
        Parse the given dictionary and fill the parsed_data member accordingly.
        Note: It is the wrapper that return the parsed_data

        :param data_dict: Dict[str, Any] Dictionary to parse
        :return: A Dict[str, Any] that represent the parsed_data member
        """

        # Parse 'positions' field
        if 'position' in data_dict:
            data_dict[self.grammar_plug[0]] = data_dict.pop('position')
        if self.grammar_plug[0] in data_dict:
            self.dirty_fields.append(self.grammar_plug[0])
            self.parsed_data[self.grammar_plug[0]] = self.parse_vector(data_dict[self.grammar_plug[0]])

    @update_wrapper()
    def update_instance(self, instance: VisualInstance) -> VisualInstance:
        """
        Update the given VisualInstance instance.

        :param instance: VisualInstance Vedo object to update with its current parsed_data values
        :return: The updated VisualInstance
        """

        # Update positions
        if self.grammar_plug[0] in self.dirty_fields:
            instance.points(self.parsed_data[self.grammar_plug[0]])
            self.dirty_fields.remove(self.grammar_plug[0])

        # Update radius
        if self.grammar_plug[1] in self.dirty_fields:
            instance.r(self.parsed_data[self.grammar_plug[1]])
            self.dirty_fields.remove(self.grammar_plug[1])

        return instance
