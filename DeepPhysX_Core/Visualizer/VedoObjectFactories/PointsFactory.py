from DeepPhysX_Core.Visualizer.VedoObjectFactories.BaseObjectFactory import *
import numpy as np
from vedo import utils


class PointsFactory(BaseObjectFactory):

    def __init__(self):
        BaseObjectFactory.__init__(self)

        self.type = "Points"

        self.grammar_plug = ['positions', 'r']
        self.grammar.extend(self.grammar_plug)
        self.default_values.update({self.grammar_plug[0]: None, self.grammar_plug[1]: 4})

    @parse_wrapper()
    def parse(self, data_dict: dict):
        pos = parse_position(data_dict=data_dict, wrap=True)
        if pos is not None:
            self.parsed_data[self.grammar_plug[0]] = pos
    @update_wrapper()
    def update_instance(self, instance):
        update_position(instance, self.parsed_data, self.dirty_fields, self.grammar_plug[0])

        if self.grammar_plug[1] in self.dirty_fields:
            instance.r(self.parsed_data[self.grammar_plug[1]])
            self.dirty_fields.remove(self.grammar_plug[1])

        return instance
