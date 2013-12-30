

from math import log, tanh

__all__ = ['normalized', 'avg', 'hyperbolic', 'tanh', 'dtanh']

def normalized (unnormalized, key=None, default=1.0) :
    ''' This yields values from a list normalized between 0.0 and 1.0. '''

    if callable(key) :
        unnormalized_values = [key(item) for item in unnormalized]
    else :
        unnormalized_values = list(unnormalized)

    floor_unnormalized_value   = min(unnormalized_values)
    ceiling_unnormalized_value = max(unnormalized_values)

    if ceiling_unnormalized_value == floor_unnormalized_value :
        for _ in unnormalized_values :
            yield default
    else :
        for value in unnormalized_values :
            yield (value - floor_unnormalized_value) / (ceiling_unnormalized_value - floor_unnormalized_value)

def avg (values) :
    values = list(values)
    return sum(values) / len(values)

def hyperbolic (y=1, z=1, base=10) :
    return 0.5 * log(y/z, base)

def dtanh (y) :
    return 1.0 - y * y

def __test__ (ut) :
    ut.assert_equal(list(normalized([0, 55, 100, 344])), [0.0, 0.15988372093023256, 0.29069767441860467, 1.0])
    ut.assert_equal(list(normalized([44, 44, 44], default=2.3)), [2.3, 2.3, 2.3])
    ut.assert_equal(list(normalized([('a', 0.56), ('b', 0.3), ('c', 0.4)], key=lambda item : item[1])),
                    [1.0, 0.0, 0.38461538461538464])

if __name__ == '__main__' :
    from nlplib.general.unittest import UnitTest
    __test__(UnitTest())

