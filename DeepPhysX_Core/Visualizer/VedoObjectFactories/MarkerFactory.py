from DeepPhysX_Core.Visualizer.VedoObjectFactories.BaseObjectFactory import *


class MarkerFactory(BaseObjectFactory):
    """
        MarkerFactory is a class that represent the data of a Marker visual object
            Description:
                MarkerFactory defines the parse and update procedures of Marker object according to Vedo.
    """
    def __init__(self):
        """
        Automatically set the attributes of a marker according to Vedo
        """
        BaseObjectFactory.__init__(self)

        self.type = "Marker"
        self.grammar_plug = ['position', 'symbol', "s", "filled"]
        self.grammar.extend(self.grammar_plug)
        self.default_values.update({self.grammar_plug[0]: [0, 0, 0], self.grammar_plug[1]: 'o',
                                    self.grammar_plug[2]: '0.1', self.grammar_plug[3]: True})

    @parse_wrapper()
    def parse(self, data_dict: ObjectDescription) -> None:
        pos = self.parse_position(data_dict=data_dict, wrap=True)
        if pos is not None:
            self.dirty_fields.append(self.grammar_plug[0])
            self.parsed_data[self.grammar_plug[0]] = pos

        for word in self.grammar_plug[1:]:
            if word in data_dict:
                self.dirty_fields.append(word)
                self.parsed_data[word] = data_dict[word]

        pass

    @update_wrapper()
    def update_instance(self, instance: VisualInstance) -> None:
        pass