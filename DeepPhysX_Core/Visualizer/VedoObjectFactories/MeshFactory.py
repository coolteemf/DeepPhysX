from DeepPhysX_Core.Visualizer.VedoObjectFactories.BaseObjectFactory import *


class MeshFactory(BaseObjectFactory):

    def __init__(self):
        BaseObjectFactory.__init__(self)

        self.type = "Mesh"
        self.number_of_dimensions = 0
        self.grammar_plug = ['inputobj', 'computeNormals']
        self.grammar.extend(self.grammar_plug)
        self.default_values.update({self.grammar_plug[0]: None, self.grammar_plug[1]: False})

    @parse_wrapper()
    def parse(self, data_dict: dict):
        # Look for position.s and cell.s to make inputobj
        if self.grammar_plug[0] not in self.parsed_data:
            inputobj = []
            # Either positions and cells have been passed as independant array or are inexistant
            if 'position' in data_dict or 'positions' in data_dict:
                inputobj.append(data_dict['position'] if 'position' in data_dict else data_dict['positions'])
                self.number_of_dimensions = inputobj[0].shape[-1]
                if 'cell' in data_dict or 'cells' in data_dict:
                    inputobj.append(data_dict['cell'] if 'cell' in data_dict else data_dict['cells'])
            self.parsed_data[self.grammar_plug[0]] = inputobj

    @update_wrapper()
    def update_instance(self, instance):
        if self.grammar_plug[0] in self.dirty_fields:
            instance.points(self.parsed_data[self.grammar_plug[0]])
            self.dirty_fields.remove(self.grammar_plug[0])
        if 'position' in self.dirty_fields:
            instance.points(self.parsed_data[self.grammar_plug[0]])
            self.dirty_fields.remove('position')
        if 'positions' in self.dirty_fields:
            instance.points(self.parsed_data[self.grammar_plug[0]])
            self.dirty_fields.remove('positions')

