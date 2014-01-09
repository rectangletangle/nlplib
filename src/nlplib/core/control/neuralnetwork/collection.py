

from nlplib.core.base import Base

__all__ = ['Array', 'Matrix', 'NLPLibArray', 'NLPLibMatrix']

class NLPLibArray (Base) :

    __slots__ = ('_values',)

    def __init__ (self, values=()) :
        self._values = tuple(values)

    def __repr__ (self, *args, **kw) :
        return super().__repr__(list(self._values), *args, **kw)

    def __iter__ (self) :
        return iter(self._values)

    def __eq__ (self, other) :
        return list(self._values) == other

    def __len__ (self) :
        return len(self._values)

class NLPLibMatrix (Base) :

    __slots__ = ('_values',)

    def __init__ (self, values=()) :
        self._values = list(values)

    def __repr__ (self, *args, **kw) :
        return super().__repr__(list(self), *args, **kw)

    def __eq__ (self, other) :
        return list(self) == other

    def __len__ (self) :
        return len(self._values)

    def __iter__ (self) :
        for row in self._values :
            yield tuple(row)

    def transpose (self) :
        return NLPLibMatrix(zip(*self._values))

    def width (self) :
        try :
            return len(self._values[0])
        except IndexError :
            return 0

    def height (self) :
        return len(self._values)

Array  = NLPLibArray
Matrix = NLPLibMatrix

try :
    import numpy
except ImportError :
    ...
else :
    __all__.extend(['NumpyArray', 'NumpyMatrix'])

    class NumpyArray (NLPLibArray) :

        __slots__ = ('_values',)

        def __init__ (self, values=()) :
            self._values = numpy.array(list(values))

    class NumpyMatrix (NLPLibMatrix) :

        __slots__ = ('_values',)

        def __init__ (self, values=()) :
            self._values = numpy.matrix(list(values))

        def __iter__ (self) :
            for row in self._values :
                yield tuple(row.flat)

        def transpose (self) :
            matrix = self.__class__.__new__(self.__class__)
            matrix._values = self._values.transpose()
            return matrix

        def width (self) :
            return self._values.shape[1]

        def height (self) :
            return self._values.shape[0]

    Array  = NumpyArray
    Matrix = NumpyMatrix

def __test__ (ut) :
    import pickle

    for array_cls in [NLPLibArray, NumpyArray, Array] :
        array = array_cls([1, 2, 3])
        ut.assert_equal(len(array), 3)
        ut.assert_equal(list(array), [1, 2, 3])
        ut.assert_equal(array, [1, 2, 3])

        array = array_cls([])
        ut.assert_equal(list(array), [])
        ut.assert_equal(array, [])

        array = array_cls([2, 9, 4])
        ut.assert_equal(pickle.loads(pickle.dumps(array)), [2, 9, 4])

    for matrix_cls in [NLPLibMatrix, NumpyMatrix, Matrix] :
        matrix = matrix_cls([(1, 2, 3), (4, 5, 6)])
        ut.assert_equal(matrix.width(), 3)
        ut.assert_equal(matrix.height(), 2)
        ut.assert_equal(len(matrix), 2)
        ut.assert_equal(list(matrix), [(1, 2, 3), (4, 5, 6)])

        matrix = matrix.transpose()
        ut.assert_equal(matrix.width(), 2)
        ut.assert_equal(matrix.height(), 3)
        ut.assert_equal(len(matrix), 3)
        ut.assert_equal(list(matrix), [(1, 4), (2, 5), (3, 6)])

        matrix = matrix.transpose()
        ut.assert_equal(matrix.width(), 3)
        ut.assert_equal(matrix.height(), 2)
        ut.assert_equal(list(matrix), [(1, 2, 3), (4, 5, 6)])

        ut.assert_equal(pickle.loads(pickle.dumps(matrix)), [(1, 2, 3), (4, 5, 6)])

if __name__ == '__main__' :
    from nlplib.general.unittest import UnitTest
    __test__(UnitTest())

