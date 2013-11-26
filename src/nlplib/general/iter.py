''' This module contains a bunch of functions which facilitate specialized iteration patterns. '''


__all__ = ['chunked', 'windowed', 'chop']

def chunked (iterable, size) :
    ''' This breaks up an iterable into multiple chunks (tuples) of a specific size. '''

    chunk = ()
    for item in iterable :
        chunk += (item,)
        if len(chunk) % size == 0 :
            yield chunk
            chunk = ()

    if len(chunk) :
        yield chunk

def windowed (iterable, size, step=1) :
    ''' This function yields a tuple of a given size then steps forward. If the step is smaller than the size, the
        function yields overlapped tuples. '''

    iterable = tuple(iterable) # Could be made more efficient, then used to make chunked.
    for i in range(0, len(iterable), step) :
        yield iterable[i:i+size]

def chop (iterable, size) :
    ''' This chops off any chunks in an iterable below a certain size. '''

    for chunk in iterable :
        try :
            chunk[size-1] # Easier to Ask for Forgiveness Than Permission, style length testing
        except IndexError :
            break
        else :
            yield chunk

def __test__ (ut) :
    ut.assert_equal(list(chunked(range(7), 3)), [(0, 1, 2), (3, 4, 5), (6,)] )
    ut.assert_equal(list(chunked(range(6), 3)), [(0, 1, 2), (3, 4, 5)]       )
    ut.assert_equal(list(chunked(range(2), 3)), [(0, 1)]                     )
    ut.assert_equal(list(chunked(range(0), 3)), []                           )

    ut.assert_equal(list(windowed(range(4), 3, 1)), [(0, 1, 2), (1, 2, 3), (2, 3), (3,)] )
    ut.assert_equal(list(windowed(range(6), 3, 3)), [(0, 1, 2), (3, 4, 5)]               )

    size = 3
    ut.assert_equal(list(chop(windowed(range(4), size, 1), size)), [(0, 1, 2), (1, 2, 3)] )
    ut.assert_equal(list(chop(chunked(range(7), size), size)),     [(0, 1, 2), (3, 4, 5)] )

if __name__ == '__main__' :
    from nlplib.general.unit_test import UnitTest
    __test__(UnitTest())

