from DeepPhysX_Core.Visualizer.VedoObjectFactories.BaseObjectFactory import *
from DeepPhysX_Core.Visualizer.VedoObjectFactories.MarkerFactory import *

class GlyphFactory(BaseObjectFactory):

    def __init__(self):
        BaseObjectFactory.__init__(self)

        self.type = "Glyph"

        self.grammar_plug = ['mesh', 'glyphObj',
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
        # Look for position.s and cell.s to make inputobj
        if self.grammar_plug[0] not in self.parsed_data:
            mesh = []
            # Either positions and cells have been passed as independant array or are inexistant
            if 'position' in data_dict or 'positions' in data_dict:
                mesh.append(data_dict['position'] if 'position' in data_dict else data_dict['positions'])
            self.parsed_data[self.grammar_plug[0]] = mesh

        self.marker_factory = MarkerFactory()
        marker_data = {}
        if self.grammar_plug[1] not in self.parsed_data:
            if 'Marker' in data_dict:
                marker_data = data_dict['Marker']
            elif "marker" in data_dict:
                marker_data = data_dict['marker']
            elif 'Markers' in data_dict:
                marker_data = data_dict['Markers']
            elif "markers" in data_dict:
                marker_data = data_dict['markers']
        else:
            marker_data = self.parsed_data[self.grammar_plug[1]]
        self.marker_factory.parse(data_dict=marker_data)
        self.parsed_data[self.grammar_plug[1]] = self.marker_factory.get_data()

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
