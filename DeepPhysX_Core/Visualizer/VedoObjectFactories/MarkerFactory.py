from DeepPhysX_Core.Visualizer.VedoObjectFactories.BaseObjectFactory import *


class MarkerFactory(BaseObjectFactory):

    def __init__(self):
        """
        MarkerFactory is a class that represent the data of a Marker visual object.
        MarkerFactory defines the parse and update procedures of Marker object according to Vedo.
        """

        BaseObjectFactory.__init__(self)

        self.type = 'Marker'
        self.grammar_plug = ['position', 'symbol', 's', 'filled']
        self.grammar.extend(self.grammar_plug)
        self.default_values.update({self.grammar_plug[0]: [0, 0, 0],
                                    self.grammar_plug[1]: 'o',
                                    self.grammar_plug[2]: '0.1',
                                    self.grammar_plug[3]: True})

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

        # Parse other fields
        for word in self.grammar_plug[1:]:
            if word in data_dict:
                self.dirty_fields.append(word)
                self.parsed_data[word] = data_dict[word]

    @update_wrapper()
    def update_instance(self, instance: VisualInstance) -> VisualInstance:
        """
        Update the given VisualInstance instance.

        :param instance: VisualInstance Vedo object to update with its current parsed_data values
        :return: The updated VisualInstance
        """

        # Todo: build a new instance
        return instance
