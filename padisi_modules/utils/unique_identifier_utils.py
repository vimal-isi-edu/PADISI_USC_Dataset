import os
from collections import Sequence
from types import FunctionType

# Helper functions for extracting unique string identifiers of python objects.


def _get_dict_str_identifier_recursive(dict_object: dict) -> str:
    """
    Helper function that extracts a string identifier of a dictionary object recursively.

    :param dict_object: Dictionary object.
    :return           : String identifier of all items of the dictionary.
    """
    assert isinstance(dict_object, dict), \
        "get_dict_identifier_recursive() Error: Variable 'dict_object' must be a dictionary."

    dict_keys = list(dict_object.keys())
    dict_keys.sort()

    identifier = []
    for dict_key in dict_keys:
        identifier.append(': '.join(["'" + dict_key + "'", get_object_str_identifier(dict_object[dict_key])]))
    identifier = ', '.join(identifier)

    return identifier


def get_dict_str_identifier(dict_object: dict) -> str:
    """
    Helper function for extracting a string identifier of a dictionary python object.

    :param dict_object: Dictionary object.
    :return           : String identifier of all items of the dictionary.
    """
    return '{' + _get_dict_str_identifier_recursive(dict_object) + '}'


def get_sequence_str_identifier(sequence_object: Sequence) -> str:
    """
    Helper function for extracting a string identifier of a Sequence python object.

    :param sequence_object: Sequence object.
    :return               : String identifier of all entries of the sequence.
    """
    return '[' + ', '.join(list(map(lambda item: get_object_str_identifier(item), sequence_object))) + "]"


def get_object_str_identifier(obj, remove_magic_keys: bool = False) -> str:
    """
    Helper function for extracting a string identifier of a python object. Supports most types of python.
    If an object does not have an __str__ method, it returns 'None'.

    :param obj              : Python object.
    :param remove_magic_keys: Do not consider magic keys (e.g., __dict__, __doc__, etc. for objects).
                  (Optional)  Default is False.
    :return                 : String identifier of the object combining all its internal variables/properties.
    """
    if hasattr(obj, '__dict__') and not isinstance(obj, FunctionType):
        obj_identifier = []
        class_properties = list(obj.__dict__.keys())
        if remove_magic_keys:
            class_properties = [item for item in class_properties if not (item.startswith('__') and
                                                                          item.endswith('__'))]
        class_properties.sort()
        for item in class_properties:
            try:
                item_value = get_object_str_identifier(getattr(obj, item))
            except Exception as e:
                print("Object of type '{}' does not have an '__str__' method: {}. Returning 'None'".
                      format(type(obj), str(e)))
                item_value = 'None'
            obj_identifier.append(':-'.join([item, item_value]))
        obj_identifier = ':-- '.join([obj.__class__.__name__, ','.join(obj_identifier)])
    else:
        try:
            if isinstance(obj, dict):
                obj_identifier = get_dict_str_identifier(obj)
            elif isinstance(obj, Sequence) and not isinstance(obj, str):
                obj_identifier = get_sequence_str_identifier(obj)
            else:
                if hasattr(obj, '__str__'):
                    obj_identifier = str(obj)
                else:
                    obj_identifier = 'None'
                if '<' in obj_identifier and '>' in obj_identifier:
                    obj_identifier = obj.__class__.__name__
                elif os.path.isdir(obj_identifier):
                    obj_identifier = 'None'
                elif os.path.isfile(obj_identifier):
                    obj_identifier = os.path.split(obj_identifier)[1]
        except Exception as e:
            print("Object of type '{}' does not have an '__str__' method: {}. Returning 'None'".
                  format(type(obj), str(e)))
            obj_identifier = 'None'

    return obj_identifier
