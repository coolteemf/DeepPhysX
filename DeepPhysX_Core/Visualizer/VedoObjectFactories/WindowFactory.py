from DeepPhysX_Core.Visualizer.VedoObjectFactories.BaseObjectFactory import *


class WindowFactory(BaseObjectFactory):

    def __init__(self):
        """
        WindowFactory is a class that represent the data of a Vedo Window.
        WindowFactory defines the parse and update procedures of a Vedo window.
        """

        BaseObjectFactory.__init__(self)

        self.type = 'Window'
        self.grammar_plug = ['objects_id', 'axes', 'sharecam', 'title', 'interactive']
        self.grammar.extend(self.grammar_plug)
        self.default_values.update({self.grammar_plug[0]: [],
                                    self.grammar_plug[1]: 2,
                                    self.grammar_plug[2]: True,
                                    self.grammar_plug[3]: 'Vedo',
                                    self.grammar_plug[4]: False})
        self.number_of_dimensions = 0

    @parse_wrapper()
    def parse(self, data_dict: ObjectDescription) -> None:
        """
        Parse the given dictionary and fill the parsed_data member accordingly.
        Note: It is the wrapper that return the parsed_data

        :param data_dict: Dict[str, Any] Dictionary to parse
        :return: A Dict[str, Any] that represent the parsed_data member
        """

        for word in self.grammar_plug:
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
        return instance
