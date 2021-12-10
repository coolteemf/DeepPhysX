from DeepPhysX_Core.Visualizer.VedoObjectFactories.BaseObjectFactory import *


class PointsFactory(BaseObjectFactory):

    def __init__(self):
        BaseObjectFactory.__init__(self)

        self.type = "Points"

        self.grammar_plug = ['inputobj', 'r']
        self.grammar.extend(self.grammar_plug)
        self.default_values.update({self.grammar_plug[0]: None, self.grammar_plug[1]: 4})

    @parse_wrapper()
    def parse(self, data_dict: dict):
        # Look for position.s and cell.s to make inputobj
        pos = []
        if self.grammar_plug[0] not in self.parsed_data:
            # Either positions and cells have been passed as independant array or are inexistant
            if 'position' in data_dict or 'positions' in data_dict:
                pos = data_dict['position'] if 'position' in data_dict else data_dict['positions']
            self.parsed_data[self.grammar_plug[0]] = pos

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

        if self.grammar_plug[1] in self.dirty_fields:
            instance.r(self.parsed_data[self.grammar_plug[1]])
            self.dirty_fields.remove(self.grammar_plug[1])