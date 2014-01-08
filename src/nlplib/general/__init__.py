''' This sub-package contains modules for dealing with non natural language processing/machine learning specific
    tasks. '''


from time import time
from functools import wraps

from nlplib.general.unittest import _logging_function

__all__ = ['composite', 'subclasses', 'timing']

class _Composite :
    def __init__ (self, key=lambda object : (), identity=lambda object : (id(object),)) :
        self.key = key
        self.identity = identity
        self.history = {}
        self.cache = {}

    def __call__ (self, function) :
        @wraps(function)
        def get (object) :
            identity = self.identity(object)
            key = self.key(object)

            hash(key) # Unhashable types shouldn't be used as a key.

            full_key = identity + key

            if self.has_changed(identity, key) :
                return self.update(object, function, identity, key, full_key)
            else :
                return self.cache[full_key]

        return property(get)

    def has_changed (self, identity, key) :
        try :
            last = self.history[identity]
        except KeyError :
            return True
        else :
            return key != last

    def update (self, object, function, identity, key, full_key) :
        self.history[identity] = key
        value = self.cache[full_key] = function(object)
        return value

def composite (*args, **kw) :
    ''' A decorator for making lazily evaluated cached read only properties. '''

    return _Composite(*args, **kw)

def subclasses (cls) :
    ''' This recursively yields all of the subclasses for a class. '''

    for subclass in cls.__subclasses__() :
        yield subclass
        yield from subclasses(subclass)

def timing (function, log=print) :
    ''' A simple decorator which prints how long a function took to return. '''

    log = _logging_function(log)

    @wraps(function)
    def timed (*args, **kw) :
        time_0 = time()
        ret = function(*args, **kw)
        time_1 = time()
        log('the function <%s> took %0.3f seconds' % (function.__name__, time_1 - time_0))
        return ret

    return timed

def __test__ (ut) :

    class Foo :
        count = 0

        def __init__ (self, z=0) :
            self.x = [5]
            self.y = [2]
            self.z = [z]

        @composite(key=lambda self : (tuple(self.x), tuple(self.y)))
        def bar (self) :
            self.count += 1
            return self.x + self.y + self.z

        @composite()
        def baz (self) :
            return 43

    foo = Foo()
    ut.assert_equal(foo.count, 0)
    ut.assert_equal(foo.bar, [5, 2, 0])
    ut.assert_equal(foo.count, 1)
    ut.assert_equal(foo.baz, 43)
    ut.assert_equal(foo.bar, [5, 2, 0])
    ut.assert_equal(foo.count, 1)
    foo.x = [3]
    ut.assert_equal(foo.bar, [3, 2, 0])
    ut.assert_equal(foo.count, 2)
    ut.assert_equal(foo.bar, [3, 2, 0])
    ut.assert_equal(foo.count, 2)
    foo.y = [4]
    ut.assert_equal(foo.bar, [3, 4, 0])
    ut.assert_equal(foo.count, 3)
    ut.assert_equal(foo.bar, [3, 4, 0])
    ut.assert_equal(foo.count, 3)

    other_foo = Foo(z=3)
    ut.assert_equal(other_foo.bar, [5, 2, 3])
    ut.assert_equal(other_foo.count, 1)
    ut.assert_equal(foo.bar, [3, 4, 0])
    ut.assert_equal(foo.count, 3)

    def cant_set (foo) :
        foo.bar = 34
    ut.assert_raises(lambda : cant_set(foo), AttributeError)

    class Baz (Foo) :
        @composite(key=lambda object : (object.x, object.y))
        def bar (self) :
            pass
    baz = Baz()
    ut.assert_raises(lambda : baz.bar, TypeError)

def __demo__ () :
    from time import sleep

    @timing
    def foo () :
        sleep(0.35)

    foo()

if __name__ == '__main__' :
    from nlplib.general.unittest import UnitTest
    __test__(UnitTest())
    __demo__()

