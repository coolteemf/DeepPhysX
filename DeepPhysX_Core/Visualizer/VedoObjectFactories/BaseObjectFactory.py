class parse_wrapper:
    def __call__(self, specialized_parse):
        def general_parse(*args, **kwargs):
            if len(args) + len(kwargs) != 2:
                return
            object_factory = args[0]
            if len(args) == 2:
                data_dict = args[1]
            else:
                (_, data_dict), *rest = kwargs.items()
            # Load all parameters directly given with the appropriate name
            object_factory.parsed_data.update(
                {word: data_dict[word] for word in object_factory.grammar if word in data_dict})
            object_factory.parsed_data["type"] = object_factory.type
            # Run the specialized parse code
            specialized_parse(self=object_factory, data_dict=data_dict)
            # Default init of all not given parameters
            for word in object_factory.grammar:
                if word not in object_factory.parsed_data:
                    object_factory.parsed_data[word] = object_factory.default_values[word]
            return object_factory.parsed_data

        return general_parse


class update_wrapper:
    def __call__(self, specialized_update):
        def general_update(*args, **kwargs):
            if len(args) + len(kwargs) != 2:  # self, instance
                return
            # First object is always self, it is passed implicitly
            object_factory = args[0]
            # Either in kwargs or 2nd argument in args but always last with update_instance definition
            if "instance" in kwargs:
                instance = kwargs["instance"]
            else:
                instance = args[-1]
            if "colormap" in object_factory.dirty_fields:
                instance.c(object_factory.parsed_data["colormap"])
                object_factory.dirty_fields.remove("colormap")
            if "scalar_field" in object_factory.dirty_fields:
                instance.addPointArray(input_array=object_factory.parsed_data["scalar_field"], name=object_factory.parsed_data["scalar_field_name"])
                object_factory.dirty_fields.remove("scalar_field")
                object_factory.dirty_fields.remove("scalar_field_name")
            if "alpha" in object_factory.dirty_fields:
                instance.alpha(object_factory.parsed_data["alpha"])
                object_factory.dirty_fields.remove("alpha")
            if "c" in object_factory.dirty_fields:
                instance.c(object_factory.parsed_data["c"])
                object_factory.dirty_fields.remove("c")

            specialized_update(self=object_factory, instance=instance)

            if object_factory.dirty_fields:  # Check for emptyness
                print(f"Update successful ! Unused field(s) : {object_factory.dirty_fields}")

            return instance
        return general_update


class BaseObjectFactory:

    def __init__(self):
        self.grammar = ['c', 'alpha', 'at', 'colormap', 'scalar_field', "scalar_field_name"]
        self.default_values = {self.grammar[0]: 'b', self.grammar[1]: 1.0, self.grammar[2]: -1, self.grammar[3]: 'jet',
                               self.grammar[4]: [], self.grammar[5]: "scalar_field"}
        self.parsed_data = {}
        self.dirty_fields = []
        self.type = None

    @parse_wrapper()
    def parse(self, data_dict: dict):
        raise NotImplementedError

    def update_dict(self, new_data: dict):
        self.dirty_fields = [*new_data.keys()]
        return self.parse(new_data)

    def get_data(self):
        return self.parsed_data

    @update_wrapper()
    def update_instance(self, instance):
        raise NotImplementedError
