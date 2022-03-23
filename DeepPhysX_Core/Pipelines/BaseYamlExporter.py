import copy

import yaml

#     :param Optional[list] excluded: List of strings. Keys that match an element of excluded will be removed from
#                                     var_dict before exporting. This is understood recursively, meaning keys in nested
#                                     dictionnaries matching an element of excluded will be removed as well. Variables
#                                     that should be exported should have a name that is not in the excluded list.

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
        keys = list(_dict.keys())
        for k in keys:
            v = _dict[k]
            if not (k[:2] == '__' and k[-2:] == '__'):
                if isinstance(v, type): #Object is just a type, not an instance
                    name = convert_repr_to_name(repr(v))
                    _dict.update({k: name})
                else:
                    if hasattr(v, '__dict__'): #Object is an instance and contains other objects
                        d = convert_dict_rec(getattr(v, '__dict__'))
                        name = convert_repr_to_name(repr(v))
                        _dict.update({k: {name: d}})
                    elif isinstance(v, tuple) and hasattr(v, '_fields'): #Object is a namedtuple (no dict)
                        d = convert_dict_rec(v._asdict())
                        name = convert_repr_to_name(repr(v))
                        _dict.update({k: {name: d}})
                    elif hasattr(v, '__iter__'): #Object contains other object but has no name
                        raise NotImplementedError("Need to be implemented")
                        #convert_dict_rec(v) #Object needs to be converted to dict before !
                    else: #Object is assumed to not contain other objects
                        pass
        return _dict
    convert_dict_rec(export_dict)
    if filename is not None:
        #Export to yaml file
        with open(filename,'w') as f:
            print(f"[BaseYamlExporter] Saving conf to {filename}.")
            yaml.dump(export_dict, f)
    else:
        return export_dict