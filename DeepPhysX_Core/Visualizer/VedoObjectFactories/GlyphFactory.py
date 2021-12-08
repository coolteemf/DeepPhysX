from DeepPhysX_Core.Visualizer.VedoObjectFactories.BaseObjectFactory import *
from DeepPhysX_Core.Visualizer.VedoObjectFactories.MarkerFactory import *
from vedo import utils
import numpy as np

class GlyphFactory(BaseObjectFactory):

    def __init__(self):
        BaseObjectFactory.__init__(self)

        self.type = "Glyph"

        self.grammar_plug = ['positions', 'glyphObj',
                             'orientationArray', 'scaleByScalar',
                             'scaleByVectorSize', 'scaleByVectorComponents',
                             'colorByScalar', 'colorByVectorSize', 'tol']
        self.grammar.extend(self.grammar_plug)
        self.default_values.update({self.grammar_plug[0]: None, self.grammar_plug[1]: None,
                                    self.grammar_plug[2]: None, self.grammar_plug[3]: False,
                                    self.grammar_plug[4]: False, self.grammar_plug[5]: False,
                                    self.grammar_plug[6]: False, self.grammar_plug[7]: False,
                                    self.grammar_plug[8]: 0})
        self.marker_factory = None

    @parse_wrapper()
    def parse(self, data_dict: dict):

        pos = parse_position(data_dict=data_dict, wrap=True)
        if pos is not None:
            self.parsed_data[self.grammar_plug[0]] = pos

        self.marker_factory = MarkerFactory()
        if 'Marker' in data_dict:
            self.marker_factory.parse(data_dict=data_dict['Marker'])
        elif "marker" in data_dict:
            self.marker_factory.parse(data_dict=data_dict['marker'])
        elif 'Markers' in data_dict:
            self.marker_factory.parse(data_dict=data_dict['Markers'])
        elif "markers" in data_dict:
            self.marker_factory.parse(data_dict=data_dict['markers'])
        elif self.grammar_plug[1] in data_dict:
            self.marker_factory.parse(data_dict=data_dict[self.grammar_plug[1]])
        self.parsed_data[self.grammar_plug[1]] = self.marker_factory.get_data()

    @update_wrapper()
    def update_instance(self, instance):
        # update_position(instance, self.parsed_data, self.dirty_fields, self.grammar_plug[0])
        pass