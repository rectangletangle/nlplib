

from math import log, tanh

__all__ = ['normalize_values', 'avg', 'hyperbolic', 'tanh', 'dtanh']

def normalize_values (unnormalized_values) :
    ''' This normalizes all of the values in a list of values between 0 and 1, assuming the value is between the floor
        and ceiling values. '''

    unnormalized_values = list(unnormalized_values)

    floor_unnormalized_value   = min(unnormalized_values)
    ceiling_unnormalized_value = max(unnormalized_values)

    for value in unnormalized_values :
        try :
            yield (value - floor_unnormalized_value) / (ceiling_unnormalized_value - floor_unnormalized_value)
        except ZeroDivisionError :
            yield 1.0

def avg (values) :
    values = list(values)
    return sum(values) / len(values)

def hyperbolic (y=1, z=1, base=10) :
    return 0.5 * log(y/z, base)

def dtanh (y) :
    return 1.0 - y * y

def __test__ (ut) :
    ut.assert_equal(list(normalize_values([0, 55, 100, 344])), [0.0, 0.15988372093023256, 0.29069767441860467, 1.0])
    ut.assert_equal(list(normalize_values([44, 44, 44])), [1.0, 1.0, 1.0])

if __name__ == '__main__' :
    from nlplib.general.unittest import UnitTest
    __test__(UnitTest())

