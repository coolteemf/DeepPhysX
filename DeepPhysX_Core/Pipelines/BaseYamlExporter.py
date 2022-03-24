import copy
import yaml

def BaseYamlExporter(filename: str=None, var_dict:dict=None):
    """
    | Exports variables in a yaml file, excluding classes, modules and functions. Additionally, variables with a name in
    | excluded will not be exported.
    :param str filename: Path to the file in which var_dict will be saved after filtering
    :param dict var_dict: Dictionnary containing the key:val pairs to be saved. Key is a variable name and val its value
    """
    export_dict = copy.deepcopy(var_dict)
    def convert_repr_to_name(repr_str: str):
        if repr_str.__contains__("<class "): #Class object, not instanciated
            return repr_str.split("<class '")[1].split("'>")[0]
        elif repr_str.__contains__(" object "): #Instance object
            return repr_str.split("<")[1].split(" ")[0]
        elif repr_str.__contains__("("): #Namedtuple object
            return repr_str.split("(")[0]
        else:
            raise ValueError(f"BaseYamlExporter: {repr_str} could not be converted to an object name.")
    def convert_dict_rec(_dict):
        if isinstance(_dict, dict):
            keys = list(_dict.keys())
            if 'excluded' in keys: #Special keyword that specify which keys should be removed
                for exclude_key in _dict['excluded']:
                    if exclude_key in _dict: _dict.pop(exclude_key) #Remove the key listed in excluded
                keys = list(_dict.keys()) #Update the keys
        elif hasattr(_dict, '__iter__'):
            keys = range(len(_dict))
        else:
            raise ValueError(f"BaseYamlExporter: encountered an object to convert which is neither a dict nor iterable.")
        for k in keys:
            v = _dict[k]
            if isinstance(k, int) or not (k[:2] == '__' and k[-2:] == '__'):
                if isinstance(v, type): #Object is just a type, not an instance
                    name = convert_repr_to_name(repr(v))
                    _dict.update({k: name})
                else:
                    if hasattr(v, '__dict__'): #Object is an instance and contains other objects, named by keys
                        d = convert_dict_rec(getattr(v, '__dict__'))
                        name = convert_repr_to_name(repr(v))
                        _dict.update({k: {name: d}})
                    elif isinstance(v, tuple) and hasattr(v, '_fields'): #Object is a namedtuple -> convert to dict
                        d = convert_dict_rec(v._asdict())
                        name = convert_repr_to_name(repr(v))
                        _dict.update({k: {name: d}})
                    elif hasattr(v, '__iter__') and not isinstance(v, str): #Object contains other objects but has no keys
                        d = convert_dict_rec(v)
                    else: #Object is assumed to not contain other objects
                        pass
        return _dict
    convert_dict_rec(export_dict)
    if filename is not None:
        #Export to yaml file
        with open(filename,'w') as f:
            print(f"[BaseYamlExporter] Saving conf to {filename}.")
            yaml.dump(export_dict, f)
    return export_dict