''' This module contains a bunch of functions which facilitate specialized iteration patterns. '''


import itertools
import collections

__all__ = ['windowed', 'chunked', 'chop', 'generates', 'truncated', 'paired', 'united', 'flattened']

def windowed (iterable, size, step=1, trail=False) :
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

        if trail :
            while len(window) :
                yield window
                window = window[step:]

def chunked (iterable, size, trail=False) :
    ''' This breaks up an iterable into multiple chunks (tuples) of a specific size. '''

    return windowed(iterable, size=size, step=size, trail=trail)

def chop (iterable, size) :
    ''' This removes any chunks at the end of an iterable, below a certain size. '''

    if size > 0 :
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

def truncated (iterable, amount) :
    ''' This allows for iteration over all but the last couple of items in an iterable. '''

    queue   = collections.deque()
    append  = queue.append
    popleft = queue.popleft

    for item in iterable :
        append(item)

        if len(queue) > amount :
            yield popleft()

def paired (iterable) :
    return windowed(iterable, size=2, step=1)

def united (paired) :
    ''' This can be used to efficiently undo the effects of the <paired> function on an iterable. '''

    paired = iter(paired)

    try :
        first, second = next(paired)
    except (StopIteration, ValueError) :
        pass
    else :
        yield first
        yield second
        for first, second in paired :
            yield second

def flattened (iterable, basecase=None) :
    if basecase is None :
        def basecase (iterable) :
            return not hasattr(iterable, '__iter__')

    if not basecase(iterable) :
        for item in iterable :
            yield from flattened(item, basecase=basecase)
    else :
        item = iterable
        yield item

def __test__ (ut) :
    ut.assert_equal(list(windowed(range(6), 3, 3)), [(0, 1, 2), (3, 4, 5)])
    ut.assert_equal(list(windowed(range(6), 1, 1)), [(0,), (1,), (2,), (3,), (4,), (5,)])

    ut.assert_equal(list(windowed(range(4), 3, 1, trail=True)), [(0, 1, 2), (1, 2, 3), (2, 3), (3,)])

    ut.assert_equal(list(chunked(range(7), 3)), [(0, 1, 2), (3, 4, 5)] )
    ut.assert_equal(list(chunked(range(6), 3)), [(0, 1, 2), (3, 4, 5)] )
    ut.assert_equal(list(chunked(range(2), 3)), []                     )
    ut.assert_equal(list(chunked(range(0), 3)), []                     )

    ut.assert_equal(list(chunked(range(4), 3, trail=True)), [(0, 1, 2), (3,)])
    ut.assert_equal(list(chunked(range(2), 3, trail=True)), [(0, 1)])

    size = 3
    ut.assert_equal(list(chop(windowed(range(4), size, 1), size)), [(0, 1, 2), (1, 2, 3)] )
    ut.assert_equal(list(chop(chunked(range(7), size), size)),     [(0, 1, 2), (3, 4, 5)] )
    ut.assert_equal(list(chop(paired([0]), 1)), [])
    ut.assert_equal(list(chop(paired([0]), 0)), [])
    ut.assert_equal(list(chop(paired([0]), -1)), [])

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

    ut.assert_equal(list(truncated([], 1)),          []              )
    ut.assert_equal(list(truncated(range(4), 1)),    [0, 1, 2]       )
    ut.assert_equal(list(truncated(range(10), 4)),   list(range(6))  )
    ut.assert_equal(list(truncated(range(10), 0)),   list(range(10)) )
    ut.assert_equal(list(truncated(range(10), -1)),  list(range(10)) )
    ut.assert_equal(list(truncated(range(10), -34)), list(range(10)) )
    ut.assert_equal(list(truncated(range(10), 34)),  []              )
    ut.assert_equal(list(truncated(range(10), 10)),  []              )

    ut.assert_equal(list(paired([])), [])
    ut.assert_equal(list(paired([0])), [])
    ut.assert_equal(list(paired(range(4))), [(0, 1), (1, 2), (2, 3)])
    ut.assert_equal(list(paired(range(5))), [(0, 1), (1, 2), (2, 3), (3, 4)])

    ut.assert_equal(list(united(paired([]))), [])
    ut.assert_equal(list(united(paired([0]))), [])
    ut.assert_equal(list(united(paired([0, 1]))), [0, 1])
    ut.assert_equal(list(united(paired(range(3)))), list(range(3)))
    ut.assert_equal(list(united(paired(range(12)))), list(range(12)))

    ut.assert_equal(list(flattened([])), [])
    ut.assert_equal(list(flattened([[], [[], []]])), [])
    ut.assert_equal(list(flattened([[[0]], [1, [2]], [3]])), [0, 1, 2, 3])
    ut.assert_equal(list(flattened([[(0,)], [(1,), [], []], (2,)],
                                   basecase=lambda iterable : isinstance(iterable, tuple))),
                    [(0,), (1,), (2,)])

if __name__ == '__main__' :
    from nlplib.general.unittest import UnitTest
    __test__(UnitTest())

