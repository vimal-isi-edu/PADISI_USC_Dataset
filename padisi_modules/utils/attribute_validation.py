from os import linesep

import numpy as np


def validate_data_attributes(data: (tuple, list, np.ndarray), classes: tuple,
                             attributes: tuple, func_name: (None, str) = None,
                             var_name: (None, str) = None, arg_index: (None, int) = None) -> np.ndarray:
    """
    Function for error checking of attributes of variables. Raises Exception (ValueError/AssertionError/RuntimeError)
    if attributes cannot be validated.
    Mimics MATLAB's function 'validateattributes()'.

    :param data      : Data to be checked - must be tuple, list or numpy array.
    :param classes   : Classes where data is going to be checked against (e.g., np.integer, np.floating, etc.).
    :param attributes: Tuple of attributes for which data should be checked against. Available attributes are:
        - Single input attributes (shape related):
            {'2d', '3d', 'scalar', 'vector', 'square', 'nonempty'}
        - Single input attributes (structure related):
            {'diag': provided array must be diagonal, i.e., all non-diagonal entries must be zero. Only applies to
                     square 2-dimensional arrays
            }
        - Single input attributes (value related - all entries must adhere to the attribute rule):
            {'diag', 'finite', 'nonnan', 'binary', 'even', 'odd', 'integer', 'real', 'positive', 'nonnegative',
            'nonzero'}
        - Single input attributes (value sequence related). Only applies to vectors:
            {'decreasing'   : values should be strictly decreasing
             'increasing'   : values should be strictly increasing
             'nondecreasing': values should be monotonically increasing
             'nonincreasing': values should be monotonically decreasing
            }
        - Double input attributes as 'attribute_name', attribute_value in sequence with other attributes:
                                   {'shape': expected shape of input data as a vector - All dimensions must match.
                                    'size' : expected number of elements in data
                                    'nrows': expected number of rows in data
                                    'ncols': expected number of columns in data
                                    'ndims': expected number of dimensions in data
                                    '>'    : all entries must be greater than the attribute value provided
                                    '>='   : all entries must be greater or equal to the attribute value provided
                                    '<'    : all entries must be smaller than the attribute value provided
                                    '<='   : all entries must be smaller or equal to the attribute value provided
    :param func_name : (Optional) - Name of function that calls the 'validate_data_attributes()' function - Used for
                       more informative error messages.
    :param var_name  : (Optional) - Name of variable that is being checked - Used for more informative error messages.
    :param arg_index : (Optional) - Index of variable that is being checked in a function's signature. Not so relevant
                       to Python.
    :return          : The provided input variable in 'data' as a numpy array. If input cannot be converted to a numpy
                       array the function raises an Exception.
    """

    def _create_error_string(_func_name, _var_name, _arg_index) -> str:
        error_str = ""
        if _func_name is not None:
            error_str += "Error using '{}'{}".format(_func_name, linesep)
        error_str += "Expected variable"
        if _arg_index is not None and _var_name is not None:
            error_str += " {}, '{}',".format(_arg_index, _var_name)
        elif _arg_index is not None:
            error_str += " {}".format(_arg_index)
        elif _var_name is not None:
            error_str += " '{}'".format(_var_name)

        return error_str

    def _get_attribute_value(attr_list: tuple, counter: int, attribute_name: str,
                             attr_classes: tuple, attr_attributes: tuple):
        assert len(attr_list) > counter + 1, \
            "{} must contain an extra entry with the attribute value.".format(attribute_name)
        attribute_value = attr_list[counter + 1]
        if np.isscalar(attribute_value):
            attribute_value = np.array([attribute_value])
        try:
            attribute_value = validate_data_attributes(attribute_value, attr_classes, attr_attributes)
        except Exception as ex:
            raise ValueError("Attribute '{}' error: ".format(attribute_name) + str(ex))
        return attribute_value, counter + 1

    def _create_attribute_error_string(err_str, name, val):
        return '{} to be an array with number of {} equal to {}.'.format(err_str, name, val)

    def _is_diagonal(array: np.ndarray):
        dummy_array = np.ones(array.shape, dtype=array.dtype)
        # Fill the diagonal of dummy matrix with 0.
        np.fill_diagonal(dummy_array, 0)

        return np.count_nonzero(np.multiply(dummy_array, array)) == 0

    assert isinstance(classes, tuple), "Valid 'classes' must be a tuple."
    assert all(tuple(map(lambda x: isinstance(x, type), classes))), "Valid 'classes' must contain types."
    assert isinstance(attributes, tuple), "Valid 'attributes' must be a tuple."
    if func_name is not None:
        assert isinstance(func_name, str), "Valid 'func_name' must be a string."
    if var_name is not None:
        assert isinstance(var_name, str), "Valid 'var_name' must be a string."
    if arg_index is not None:
        assert isinstance(arg_index, int) and arg_index >= 0, "Valid 'arg_index' must be a positive integer."

    assert isinstance(data, (tuple, list, np.ndarray)), "{} to be a tuple, a list or a numpy array.". \
        format(_create_error_string(func_name, var_name, arg_index))
    if isinstance(data, (tuple, list)):
        try:
            data = np.array(data)
        except ValueError:
            raise

    if len(classes) > 0 and not np.any(tuple(map(lambda x: np.issubdtype(data.dtype, x), classes))):
        raise ValueError("{} to be one of these types: {}{}{}{}{} Instead, its type was {}".format(
            _create_error_string(func_name, var_name, arg_index), linesep, linesep,
            ', '.join(tuple(map(lambda x: "'" + np.dtype(x).name + "'", classes))),
            linesep, linesep, "'" + data.dtype.name + "'"))

    attr_counter = 0
    error_string = _create_error_string(func_name, var_name, arg_index)
    while attr_counter < len(attributes):
        attr_name = attributes[attr_counter]
        assert isinstance(attr_name, str), "Valid 'attributes' must be strings."
        attr_name = attr_name.lower()

        if attr_name == '2d':
            assert data.ndim <= 2, '{} to be two-dimensional.'.format(error_string)
        elif attr_name == '3d':
            assert data.ndim <= 3, '{} to be three-dimensional.'.format(error_string)
        elif attr_name == 'scalar':
            assert data.size == 1, '{} to be a scalar.'.format(error_string)
        elif attr_name == 'vector':
            assert data.ndim == 1, '{} to be a one-dimensional.'.format(error_string)
        elif attr_name == 'shape':
            attr_val, attr_counter = _get_attribute_value(attributes, attr_counter, 'shape',
                                                          (np.integer, np.floating), ('vector', 'real'))
            temp_shape = np.array(data.shape)
            nan_idx = np.isnan(attr_val)
            if np.any(nan_idx):
                try:
                    validate_data_attributes(attr_val[np.logical_not(nan_idx)], (),
                                             ('real', 'nonnan', 'finite', 'integer', 'nonnegative'))
                except Exception as e:
                    raise ValueError("Attribute 'shape' error: " + str(e))

                temp_shape = temp_shape.astype(np.floating)
                temp_shape[nan_idx[:np.minimum(data.ndim, nan_idx.size)]] = np.nan

            assert temp_shape.size == attr_val.size, "{} to have {} dimensions but it has {} dimensions.". \
                format(error_string, attr_val.size, temp_shape.size)
            assert tuple(temp_shape[np.logical_not(nan_idx)]) == tuple(attr_val[np.logical_not(nan_idx)]), \
                '{} to be of shape {}, but it is of shape ' \
                '{}.'.format(error_string,
                             'x'.join(tuple(map(lambda x: 'NaN' if np.isnan(x) else str(int(x)), attr_val))),
                             'x'.join(tuple(map(lambda x: 'NaN' if np.isnan(x) else str(int(x)), temp_shape))))
        elif attr_name == 'size':
            attr_val, attr_counter = _get_attribute_value(attributes, attr_counter, 'size', (np.integer,),
                                                          ('scalar', 'real', 'nonnan', 'finite', 'nonnegative',))
            assert data.size == attr_val[0], '{}'. \
                format(_create_attribute_error_string(error_string, 'elements', attr_val[0]))
        elif attr_name == 'ncols':
            attr_val, attr_counter = _get_attribute_value(attributes, attr_counter, 'ncols', (np.integer,),
                                                          ('scalar', 'real', 'nonnan', 'finite', 'positive'))
            assert data.ndim >= 2 and data.shape[1] == attr_val[0], '{}'. \
                format(_create_attribute_error_string(error_string, 'columns', attr_val[0]))
        elif attr_name == 'nrows':
            attr_val, attr_counter = _get_attribute_value(attributes, attr_counter, 'nrows', (np.integer,),
                                                          ('scalar', 'real', 'nonnan', 'finite', 'positive'))
            assert data.ndim >= 1 and data.shape[0] == attr_val[0], '{}' \
                .format(_create_attribute_error_string(error_string, 'rows', attr_val[0]))
        elif attr_name == 'ndims':
            attr_val, attr_counter = _get_attribute_value(attributes, attr_counter, 'ndims', (np.integer,),
                                                          ('scalar', 'real', 'nonnan', 'finite', 'positive'))
            assert data.ndim == attr_val[0], '{}'. \
                format(_create_attribute_error_string(error_string, 'dimensions', attr_val[0]))
        elif attr_name == 'square':
            assert data.ndim == 2 and data.shape[0] == data.shape[1], \
                '{} to be a two-dimensional square array.'.format(error_string)
        elif attr_name == 'diag':
            assert data.ndim == 2 and data.shape[0] == data.shape[1] and _is_diagonal(data), \
                '{} to be a diagonal two-dimensional array.'.format(error_string)
        elif attr_name == 'nonempty':
            assert data.size > 0, '{} to be nonempty.'.format(error_string)
        elif attr_name == '>':
            attr_val, attr_counter = _get_attribute_value(attributes, attr_counter, '>', (np.integer, np.floating),
                                                          ('scalar', 'real', 'nonnan'))
            assert np.all(np.logical_and(np.isreal(data), data > attr_val[0])), \
                '{} to be an array with all values > {}.'.format(error_string, attr_val[0])
        elif attr_name == '>=':
            attr_val, attr_counter = _get_attribute_value(attributes, attr_counter, '>=', (np.integer, np.floating),
                                                          ('scalar', 'real', 'nonnan'))
            assert np.all(np.logical_and(np.isreal(data), data >= attr_val[0])), \
                '{} to be an array with all values >= {}.'.format(error_string, attr_val[0])
        elif attr_name == '<':
            attr_val, attr_counter = _get_attribute_value(attributes, attr_counter, '<', (np.integer, np.floating),
                                                          ('scalar', 'real', 'nonnan'))
            assert np.all(np.logical_and(np.isreal(data), data < attr_val[0])), \
                '{} to be an array with all values < {}.'.format(error_string, attr_val[0])
        elif attr_name == '<=':
            attr_val, attr_counter = _get_attribute_value(attributes, attr_counter, '<=', (np.integer, np.floating),
                                                          ('scalar', 'real', 'nonnan'))
            assert np.all(np.logical_and(np.isreal(data), data <= attr_val[0])), \
                '{} to be an array with all values <= {}.'.format(error_string, attr_val[0])
        elif attr_name == 'finite':
            assert np.all(np.isfinite(data)), '{} to be finite.'.format(error_string)
        elif attr_name == 'nonnan':
            assert not np.any(np.isnan(data)), '{} to be non-NaN.'.format(error_string)
        elif attr_name == 'binary':
            assert np.all(np.isreal(data)) and np.all(np.unique(data) == np.array([0, 1])), \
                '{} to be binary.'.format(error_string)
        elif attr_name == 'even':
            assert np.all(np.isreal(data)) and np.count_nonzero(np.mod(data, 2)) == 0, \
                '{} to be even.'.format(error_string)
        elif attr_name == 'odd':
            assert np.all(np.isreal(data)) and np.all(np.mod(data, 2) == 1), \
                '{} to be odd.'.format(error_string)
        elif attr_name == 'integer':
            assert np.all(np.round(data) == data), '{} to be integer-valued.'.format(error_string)
        elif attr_name == 'real':
            assert np.all(np.isreal(data)), '{} to be real.'.format(error_string)
        elif attr_name == 'nonnegative':
            assert np.all(np.logical_and(np.isreal(data), data >= 0)), '{} to be nonnegative.'.format(error_string)
        elif attr_name == 'nonzero':
            assert np.all(np.isreal(data)) and np.count_nonzero(data) == data.size, \
                '{} to be nonzero.'.format(error_string)
        elif attr_name == 'positive':
            assert np.all(np.logical_and(np.isreal(data), data > 0)), \
                '{} to be positive.'.format(error_string)
        elif attr_name == 'decreasing':
            assert data.ndim == 1 and np.all(np.isreal(data)) and \
                   np.all([data[i+1] < data[i] for i in range(data.size-1)]), '{} to be strictly ' \
                                                                              'decreasing.'.format(error_string)
        elif attr_name == 'increasing':
            assert data.ndim == 1 and np.all(np.isreal(data)) and \
                   np.all([data[i+1] > data[i] for i in range(data.size-1)]), '{} to be strictly ' \
                                                                              'increasing.'.format(error_string)
        elif attr_name == 'nondecreasing':
            assert data.ndim == 1 and np.all(np.isreal(data)) and \
                   np.all([data[i+1] >= data[i] for i in range(data.size - 1)]), '{} to be monotonically ' \
                                                                                 'increasing.'.format(error_string)
        elif attr_name == 'nonincreasing':
            assert data.ndim == 1 and np.all(np.isreal(data)) and \
                   np.all([data[i+1] <= data[i] for i in range(data.size - 1)]), '{} to be monotonically ' \
                                                                                 'decreasing.'.format(error_string)
        else:
            raise RuntimeError("Unknown attribute '{}' was provided.".format(attr_name))

        attr_counter += 1

    return data
