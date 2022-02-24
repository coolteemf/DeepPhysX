from DeepPhysX_Core.Visualizer.VedoObjectFactories.BaseObjectFactory import *
from DeepPhysX_Core.Visualizer.VedoObjectFactories.MarkerFactory import MarkerFactory


class GlyphFactory(BaseObjectFactory):

    def __init__(self):
        """
        GlyphFactory is a class that represent the data of a Glyph visual object.
        GlyphFactory defines the parse and update procedures of Glyph object according to Vedo.
        """

        BaseObjectFactory.__init__(self)

        self.type = 'Glyph'
        self.grammar_plug = ['positions', 'glyphObj', 'orientationArray', 'scaleByScalar', 'scaleByVectorSize',
                             'scaleByVectorComponents', 'colorByScalar', 'colorByVectorSize', 'tol']
        self.grammar.extend(self.grammar_plug)
        self.default_values.update({self.grammar_plug[0]: None,
                                    self.grammar_plug[1]: None,
                                    self.grammar_plug[2]: None,
                                    self.grammar_plug[3]: False,
                                    self.grammar_plug[4]: False,
                                    self.grammar_plug[5]: False,
                                    self.grammar_plug[6]: False,
                                    self.grammar_plug[7]: False,
                                    self.grammar_plug[8]: 0})

        self.marker_factory: MarkerFactory = MarkerFactory()

    @parse_wrapper()
    def parse(self, data_dict: ObjectDescription) -> None:
        """
        Parse the given dictionary and fill the parsed_data member accordingly.
        Note: It is the wrapper that return the parsed_data

        :param data_dict: Dict[str, Any] Dictionary to parse
        :return: A Dict[str, Any] that represent the parsed_data member
        """

        # Parse 'positions' field
        if 'position' in data_dict:
            data_dict[self.grammar_plug[0]] = data_dict.pop('position')
        if self.grammar_plug[0] in data_dict:
            self.dirty_fields.append(self.grammar_plug[0])
            self.parsed_data[self.grammar_plug[0]] = self.parse_vector(data_dict[self.grammar_plug[0]])

        # Parse Marker fields
        self.marker_factory = MarkerFactory()
        for name in ['Marker', 'Markers', 'marker', 'markers']:
            if name in data_dict:
                self.dirty_fields.append(self.grammar_plug[1])
                self.marker_factory.parse(data_dict=data_dict[name])
        if self.grammar_plug[1] in self.dirty_fields:
            self.parsed_data[self.grammar_plug[1]] = self.marker_factory.get_data()

    @update_wrapper()
    def update_instance(self, instance: VisualInstance) -> VisualInstance:
        """
        Update the given VisualInstance instance.

        :param instance: VisualInstance Vedo object to update with its current parsed_data values
        :return: The updated VisualInstance
        """

        # Todo: create a new instance
        return instance
