''' This module contains a bunch of functions which facilitate specialized iteration patterns. '''

# todo : make windowed trail or not, so chop isn't needed

import itertools

__all__ = ['windowed', 'chunked', 'chop', 'generates']

def windowed (iterable, size, step=1) :
    ''' This function yields a tuple of a given size, then steps forward. If the step is smaller than the size, the
        function yields "overlapped" tuples. '''

    if size == 1 and step == 1 :
        # A more efficient implementation for this particular special case.
        for item in iterable :
            yield (item,)
    else :
        window = ()
        for item in iterable :
            window += (item,)
            if len(window) == size :
                yield window
                window = window[step:]

        while len(window) :
            yield window
            window = window[step:]

def chunked (iterable, size) :
    ''' This breaks up an iterable into multiple chunks (tuples) of a specific size. '''

    return windowed(iterable, size=size, step=size)

def chop (iterable, size) :
    ''' This chops off any chunks in an iterable below a certain size. '''

    for chunk in iterable :
        try :
            chunk[size-1] # Easier to Ask for Forgiveness Than Permission, style length testing
        except IndexError :
            break
        else :
            yield chunk

def generates (generator) :
    ''' If a generator doesn't generate anything this returns <None>, otherwise it returns an equivalent generator. '''

    iterable = iter(generator)

    try :
        first = next(iterable)
    except StopIteration :
        return None
    else :
        return itertools.chain((first,), iterable)

def __test__ (ut) :
    ut.assert_equal(list(chunked(range(7), 3)), [(0, 1, 2), (3, 4, 5), (6,)] )
    ut.assert_equal(list(chunked(range(6), 3)), [(0, 1, 2), (3, 4, 5)]       )
    ut.assert_equal(list(chunked(range(2), 3)), [(0, 1)]                     )
    ut.assert_equal(list(chunked(range(0), 3)), []                           )

    ut.assert_equal(list(windowed(range(4), 3, 1)), [(0, 1, 2), (1, 2, 3), (2, 3), (3,)] )
    ut.assert_equal(list(windowed(range(6), 3, 3)), [(0, 1, 2), (3, 4, 5)]               )
    ut.assert_equal(list(windowed(range(6), 1, 1)), [(0,), (1,), (2,), (3,), (4,), (5,)] )

    size = 3
    ut.assert_equal(list(chop(windowed(range(4), size, 1), size)), [(0, 1, 2), (1, 2, 3)] )
    ut.assert_equal(list(chop(chunked(range(7), size), size)),     [(0, 1, 2), (3, 4, 5)] )

    def generates_something () :
        i = 0
        while True :
            yield i
            i += 1

    def generates_nothing () :
        for _ in () :
            yield

    ut.assert_true(generates(generates_nothing()) is None)
    ut.assert_doesnt_raise(lambda : next(generates(generates_something())), StopIteration)
    ut.assert_doesnt_raise(lambda : next(generates_something()), StopIteration)
    ut.assert_raises(lambda : next(generates_nothing()), StopIteration)

if __name__ == '__main__' :
    from nlplib.general.unit_test import UnitTest
    __test__(UnitTest())

