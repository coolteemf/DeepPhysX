from DeepPhysX_Core.Visualizer.VedoObjectFactories.BaseObjectFactory import *


class PointsFactory(BaseObjectFactory):
    """
    PointsFactory is a class that represent the data of a Points visual object
        Description:
            PointsFactory defines the parse and update procedures of Points object according to Vedo.
    """
    def __init__(self):
        """
        Automatically set the attributes of a points object according to Vedo
        """
        BaseObjectFactory.__init__(self)

        self.type = "Points"

        self.grammar_plug = ['positions', 'r']
        self.grammar.extend(self.grammar_plug)
        self.default_values.update({self.grammar_plug[0]: None, self.grammar_plug[1]: 4})

    @parse_wrapper()
    def parse(self, data_dict: ObjectDescription) -> None:
        pos = self.parse_position(data_dict=data_dict, wrap=True)
        if pos is not None:
            self.dirty_fields.append(self.grammar_plug[0])
            self.parsed_data[self.grammar_plug[0]] = pos

    @update_wrapper()
    def update_instance(self, instance: VisualInstance) -> None:
        self.update_position(instance)

        if self.grammar_plug[1] in self.dirty_fields:
            instance.r(self.parsed_data[self.grammar_plug[1]])
            self.dirty_fields.remove(self.grammar_plug[1])

        return instance
