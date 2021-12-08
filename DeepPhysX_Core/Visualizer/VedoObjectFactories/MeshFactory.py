from DeepPhysX_Core.Visualizer.VedoObjectFactories.BaseObjectFactory import *
import numpy as np
from vedo import utils

class MeshFactory(BaseObjectFactory):

    def __init__(self):
        BaseObjectFactory.__init__(self)

        self.type = "Mesh"
        self.number_of_dimensions = 0
        self.grammar_plug = ['positions', 'cells', 'computeNormals']
        self.grammar.extend(self.grammar_plug)
        self.default_values.update({self.grammar_plug[0]: None, self.grammar_plug[1]: None, self.grammar_plug[2]: False})

    @parse_wrapper()
    def parse(self, data_dict: dict):
        pos = parse_position(data_dict=data_dict, wrap=True)
        if pos is not None:
            self.parsed_data[self.grammar_plug[0]] = pos

        if 'cell' in data_dict or 'cells' in data_dict:
            self.parsed_data[self.grammar_plug[1]] = np.array(data_dict['cell'] if 'cell' in data_dict else data_dict['cells'], dtype=int)

    @update_wrapper()
    def update_instance(self, instance):
        update_position(instance, self.parsed_data, self.dirty_fields, self.grammar_plug[0])


