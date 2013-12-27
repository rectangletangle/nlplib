

from functools import wraps
from itertools import chain

__all__ = ['composite', 'pretty_truncate', 'pretty_float', 'nonliteral_representation', 'literal_representation',
           'all_subclasses']

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

def pretty_truncate (string, cutoff=100, tail='...') :
    ''' This limits the length of a string in a somewhat aesthetically pleasing fashion. '''

    try :
        return string[:cutoff].rstrip() + tail if len(string) > cutoff else string
    except TypeError :
        # This happens if <cutoff> is None
        return string

def pretty_float (value) :
    return round(float(value), 5)

def _representation_args_and_kw (*args, **kw) :
    # <sorted> is used to make the keyword argument order deterministic.
    return chain((repr(arg) for arg in args),
                 sorted(str(name) + '=' + repr(value) for name, value in kw.items()))

def nonliteral_representation (object, *args, **kw) :
    ''' This is a function for generating representations (<repr>) that can't be used as literal Python code. '''

    representation = chain((object.__class__.__name__,),
                           _representation_args_and_kw(*args, **kw))

    return '<' + ' '.join(representation) + '>'

def literal_representation (object, *args, **kw) :
    ''' This is a function for generating representations (<repr>) that can be used as literal Python code. '''

    return '{class_name}({args_and_kw})'.format(class_name=object.__class__.__name__,
                                                args_and_kw=', '.join(_representation_args_and_kw(*args, **kw)))

def all_subclasses (cls) :
    for subclass in cls.__subclasses__() :
        yield subclass
        yield from all_subclasses(subclass)

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

    ut.assert_equal(pretty_truncate('hello world', 6, '...'), 'hello...')
    ut.assert_equal(pretty_truncate('hello world', 10000, '...'), 'hello world')

    object, args, kw = ('', (1, 2, 3), {'foo' : 35, 'bar' : 40})

    ut.assert_equal(nonliteral_representation(object, *args, **kw), '<str 1 2 3 bar=40 foo=35>')
    ut.assert_equal(nonliteral_representation(object, *args), '<str 1 2 3>')
    ut.assert_equal(nonliteral_representation(object, **kw), '<str bar=40 foo=35>')
    ut.assert_equal(nonliteral_representation(object), '<str>')

    ut.assert_equal(literal_representation(object, *args, **kw), 'str(1, 2, 3, bar=40, foo=35)')
    ut.assert_equal(literal_representation(object, *args), 'str(1, 2, 3)')
    ut.assert_equal(literal_representation(object, **kw), 'str(bar=40, foo=35)')
    ut.assert_equal(literal_representation(object), 'str()')

if __name__ == '__main__' :
    from nlplib.general.unittest import UnitTest
    __test__(UnitTest())

    def comp () :
        name = 'bar'
        isset = False
        normvalue = None

        def wrapper (function) :
            compvalue = None

            def getter (self) :
                return compvalue

            def setter (self, value) :
                nonlocal compvalue, isset
                compvalue = value
                isset = True

            def norm (*args, **kw) :
                if isset or normvalue is None :
                    nonlocal isset, normvalue
                    isset = False
                    normvalue = function(*args, **kw)
                    return normvalue
                else :
                    return normvalue

            def _wrapper (self, *args, **kw) :

                compvalue = getattr(self, name)
                setattr(self.__class__, name, property(getter, setter))

                # todo : check if already prop, so this can be nested
                setattr(self.__class__, function.__name__, property(norm))

                return function(self, *args, **kw)

            return property(_wrapper)

        return wrapper

    class Foo :
        def __init__ (self) :
            self.bar = 32

        @comp()
        def foo (self) :
            print(24)
            return 'something'

    foo = Foo()

    foo.bar = 'set 0'
    print(foo.bar)
    print(foo.foo)
    print(foo.foo)
    foo.bar = 'set 1'
    print(foo.bar)
    print(foo.foo)
    print(foo.foo)














