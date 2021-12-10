from DeepPhysX_Core.Visualizer.VedoObjectFactories.BaseObjectFactory import *


class MarkerFactory(BaseObjectFactory):

    def __init__(self):
        BaseObjectFactory.__init__(self)

        self.type = "Marker"
        self.grammar_plug = ['symbol', 'pos', "s", "filled"]
        self.grammar.extend(self.grammar_plug)
        self.default_values.update({self.grammar_plug[0]: 'o', self.grammar_plug[1]: (0, 0, 0),
                                    self.grammar_plug[2]: '0.1', self.grammar_plug[3]: True})

    @parse_wrapper()
    def parse(self, data_dict: dict):
        for word in self.grammar_plug:
            if word in data_dict:
                self.parsed_data[word] = data_dict[word]
        pass

    @update_wrapper()
    def update_instance(self, instance):
        pass