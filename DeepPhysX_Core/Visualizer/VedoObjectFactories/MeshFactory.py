from DeepPhysX_Core.Visualizer.VedoObjectFactories.BaseObjectFactory import *
import numpy as np


class MeshFactory(BaseObjectFactory):
    """
    MeshFactory is a class that represent the data of a Mesh visual object
        Description:
            MeshFactory defines the parse and update procedures of Mesh object according to Vedo.
    """
    def __init__(self):
        """
        Automatically set the attributes of a mesh according to Vedo
        """
        BaseObjectFactory.__init__(self)

        self.type = "Mesh"
        self.number_of_dimensions = 0
        self.grammar_plug = ['positions', 'cells', 'computeNormals']
        self.grammar.extend(self.grammar_plug)
        self.default_values.update({self.grammar_plug[0]: None, self.grammar_plug[1]: None, self.grammar_plug[2]: False})

    @parse_wrapper()
    def parse(self, data_dict: ObjectDescription) -> None:
        pos = self.parse_position(data_dict=data_dict, wrap=True)
        if pos is not None:
            self.dirty_fields.append(self.grammar_plug[0])
            self.parsed_data[self.grammar_plug[0]] = pos

        if 'cell' in data_dict or 'cells' in data_dict:
            self.dirty_fields.append(self.grammar_plug[1])
            self.parsed_data[self.grammar_plug[1]] = np.array(data_dict['cell'] if 'cell' in data_dict else data_dict['cells'], dtype=int)

    @update_wrapper()
    def update_instance(self, instance: VisualInstance) -> None:
        self.update_position(instance)


