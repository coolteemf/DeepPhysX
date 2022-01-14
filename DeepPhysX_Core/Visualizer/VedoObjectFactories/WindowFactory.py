from DeepPhysX_Core.Visualizer.VedoObjectFactories.BaseObjectFactory import *


class WindowFactory(BaseObjectFactory):
    """
    WindowFactory is a class that represent the data of a Vedo Window
        Description:
            WindowFactory defines the parse and update procedures of a Vedo window
    """
    def __init__(self):
        """
        Automatically set the attributes of a Vedo window
        """
        BaseObjectFactory.__init__(self)

        self.type = "Window"
        self.number_of_dimensions = 0
        self.grammar_plug = ['objects_id', 'axes', 'sharecam', 'title', 'interactive']
        self.grammar.extend(self.grammar_plug)
        self.default_values.update({self.grammar_plug[0]: [], self.grammar_plug[1]: 2, self.grammar_plug[2]: True, self.grammar_plug[3]: 'Vedo', self.grammar_plug[4]: False})

    @parse_wrapper()
    def parse(self, data_dict: ObjectDescription) -> None:
        for word in self.grammar_plug:
            if word in data_dict:
                self.parsed_data[word] = data_dict[word]

    @update_wrapper()
    def update_instance(self, instance: VisualInstance) -> None:
        pass