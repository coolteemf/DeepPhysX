from DeepPhysX_Core.Visualizer.VedoObjectFactories.BaseObjectFactory import *


class ArrowsFactory(BaseObjectFactory):

    def __init__(self):
        """
        ArrowsFactory is a class that represent the data of Arrows visual object.
        It defines the parse and update procedures of Arrows object according to Vedo.
        """

        BaseObjectFactory.__init__(self)

        self.type = "Arrows"
        self.grammar_plug = ['positions', 'vectors', 'res']
        self.grammar.extend(self.grammar_plug)
        self.default_values.update({self.grammar_plug[0]: None,
                                    self.grammar_plug[1]: None,
                                    self.grammar_plug[2]: 12})

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

        # Parse 'vectors' field
        if 'vector' in data_dict:
            data_dict[self.grammar_plug[1]] = data_dict.pop('vector')
        if self.grammar_plug[1] in data_dict:
            self.dirty_fields.append(self.grammar_plug[1])
            self.parsed_data[self.grammar_plug[1]] = self.parse_vector(data_dict[self.grammar_plug[1]])

    @update_wrapper()
    def update_instance(self, instance: VisualInstance) -> VisualInstance:
        """
        Update the given VisualInstance instance.

        :param instance: VisualInstance Vedo object to update with its current parsed_data values
        :return: The updated VisualInstance
        """

        # Reset dirty fields
        self.dirty_fields = []

        # Create new Arrows instance
        return Arrows(startPoints=self.parsed_data[self.grammar_plug[0]],
                      endPoints=self.parsed_data[self.grammar_plug[0]] + self.parsed_data[self.grammar_plug[1]],
                      res=self.parsed_data[self.grammar_plug[2]],
                      c=self.parsed_data[self.grammar[0]],
                      alpha=self.parsed_data[self.grammar[1]])
