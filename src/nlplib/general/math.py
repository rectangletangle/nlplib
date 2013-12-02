

from math import log, tanh

__all__ = ['normalize_value', 'normalize_values', 'avg', 'hyperbolic', 'tanh', 'dtanh']

def normalize_value (value, floor, ceiling) :
    ''' This normalizes a value between 0 and 1, assuming the value is between the floor and ceiling values. '''

    try :
        return (value - floor) / (ceiling - floor)
    except ZeroDivisionError :
        return 0.0

def normalize_values (unnormalized_values) :
    unnormalized_values = list(unnormalized_values)

    floor_unnormalized_value   = min(unnormalized_values)
    ceiling_unnormalized_value = max(unnormalized_values)

    return (normalize_value(value, floor_unnormalized_value, ceiling_unnormalized_value)
            for value in unnormalized_values)

def avg (values) :
    values = list(values)
    return sum(values) / len(values)

def hyperbolic (y=1, z=1, base=10) :
    return 0.5 * log(y/z, base)

def dtanh (y) :
    return 1.0 - y * y

def __test__ (ut) :
    ut.assert_equal(normalize_value(55, 0, 100), 0.55)

if __name__ == '__main__' :
    from nlplib.general.unit_test import UnitTest
    __test__(UnitTest())

