from DeepPhysX_Core.Visualizer.VedoObjectFactories.BaseObjectFactory import *


class WindowFactory(BaseObjectFactory):

    def __init__(self):
        BaseObjectFactory.__init__(self)

        self.type = "Window"
        self.number_of_dimensions = 0
        self.grammar_plug = ['objects_id', 'axes', 'sharecam', 'title', 'interactive']
        self.grammar.extend(self.grammar_plug)
        self.default_values.update({self.grammar_plug[0]: [], self.grammar_plug[1]: 2, self.grammar_plug[2]: True, self.grammar_plug[3]: 'Vedo', self.grammar_plug[4]: False})

    @parse_wrapper()
    def parse(self, data_dict: dict):
        for word in self.grammar_plug:
            if word in data_dict:
                self.parsed_data[word] = data_dict[word]

    @update_wrapper()
    def update_instance(self, instance):
        pass