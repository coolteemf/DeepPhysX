from DeepPhysX_Core.Visualizer.VedoObjectFactories.BaseObjectFactory import *
import numpy as np
from vedo import utils

class MarkerFactory(BaseObjectFactory):

    def __init__(self):
        BaseObjectFactory.__init__(self)

        self.type = "Marker"
        self.grammar_plug = ['position', 'symbol', "s", "filled"]
        self.grammar.extend(self.grammar_plug)
        self.default_values.update({self.grammar_plug[0]: [0, 0, 0], self.grammar_plug[1]: 'o',
                                    self.grammar_plug[2]: '0.1', self.grammar_plug[3]: True})

    @parse_wrapper()
    def parse(self, data_dict: dict):

        pos = parse_position(data_dict=data_dict, wrap=True)
        if pos is not None:
            self.parsed_data[self.grammar_plug[0]] = pos

        for word in self.grammar_plug[1:]:
            if word in data_dict:
                self.parsed_data[word] = data_dict[word]

        pass

    @update_wrapper()
    def update_instance(self, instance):
        # print(self.parsed_data)
        # update_position(instance, self.parsed_data, self.dirty_fields, self.grammar_plug[0])
        pass