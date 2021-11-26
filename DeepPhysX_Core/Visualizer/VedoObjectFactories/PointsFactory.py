from DeepPhysX_Core.Visualizer.VedoObjectFactories.BaseObjectFactory import *
import numpy as np
from vedo import utils

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
        return instance
