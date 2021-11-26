from DeepPhysX_Core.Visualizer.VedoObjectFactories.BaseObjectFactory import *
import numpy as np
from vedo import utils

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

        inputobj = []
        # Look for position.s and cell.s to make inputobj
        pos = None

        # Either positions and cells have been passed as independant array or are inexistant
        if 'position' in data_dict:
            pos = data_dict['position']
        elif 'positions' in data_dict:
            pos = data_dict['positions']
        elif self.grammar_plug[0] in data_dict:
            pos = data_dict[self.grammar_plug[0]]

        if utils.isSequence(pos): # passing point coords
            if not utils.isSequence(pos[0]):
                pos = [pos]
            n = len(pos)

            if n == 3:  # assume pos is in the format [all_x, all_y, all_z]
                if utils.isSequence(pos[0]) and len(pos[0]) > 3:
                    pos = np.stack((pos[0], pos[1], pos[2]), axis=1)
            elif n == 2:  # assume pos is in the format [all_x, all_y, 0]
                if utils.isSequence(pos[0]) and len(pos[0]) > 3:
                    pos = np.stack((pos[0], pos[1], np.zeros(len(pos[0]))), axis=1)

            if n and len(pos[0]) == 2: # make it 3d
                pos = np.c_[np.array(pos), np.zeros(len(pos))]

            inputobj.append(pos)

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

