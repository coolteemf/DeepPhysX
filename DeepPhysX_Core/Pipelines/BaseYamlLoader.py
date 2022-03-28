import yaml
import importlib

def BaseYamlLoader(filename: str):
    with open(filename, 'r') as f:
        loaded_dict = yaml.load(f, yaml.Loader)

    def convert_type_str_to_type(name: str):
        module = importlib.import_module('.'.join(name.split('.')[:-1]))
        object_name_in_module = name.split('.')[-1]
        return getattr(module, object_name_in_module)

    def convert_variables(var_container):
        var_container_type = type(var_container)
        if isinstance(var_container, dict):
            keys = list(var_container.keys())
        elif isinstance(var_container, (tuple, list, set)):  # Is not a dict but is iterable.
            keys = range(len(var_container))
            var_container = list(var_container) #Allows to change elements in var_container
        else:
            raise ValueError(f"BaseYamlLoader: encountered an object to convert which is not a dict, tuple or list.")
        for k in keys:
            v = var_container[k]
            # Detection of a type object that was converted to str
            if isinstance(v,dict) and len(v) == 1 and 'type' in v and isinstance(v['type'], str):
                new_val = convert_type_str_to_type(v['type'])
            elif hasattr(v, '__iter__') and not isinstance(v, str):  # Object contains other objects
                new_val = convert_variables(v)
            else:
                new_val = v
            var_container[k] = new_val
        if var_container_type in (tuple, set):
            var_container = var_container_type(var_container) #Convert back to original type
        return var_container
    convert_variables(loaded_dict)
    return loaded_dict

