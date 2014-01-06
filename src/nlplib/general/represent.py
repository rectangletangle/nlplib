''' This module covers the representation of objects when they're being printed to the screen (with the <__repr__>
    magic method). This makes reading outputs slightly more pleasant. '''


from itertools import chain

__all__ = ['represented_nonliterally', 'represented_literally', 'pretty_truncate', 'pretty_float', 'pretty_ellipsis']

def _representation_args_and_kw (*args, **kw) :
    # <sorted> is used to make the keyword argument order deterministic.
    return chain((repr(arg) for arg in args),
                 sorted(str(name) + '=' + repr(value) for name, value in kw.items()))

def represented_nonliterally (object, *args, **kw) :
    ''' This is a function for generating representations (<repr>) that can't be used as literal Python code. '''

    representation = chain((object.__class__.__name__,),
                           _representation_args_and_kw(*args, **kw))

    return '<' + ' '.join(representation) + '>'

def represented_literally (object, *args, **kw) :
    ''' A function for generating representations (<repr>) that can be used as literal Python code. '''

    return '{class_name}({args_and_kw})'.format(class_name=object.__class__.__name__,
                                                args_and_kw=', '.join(_representation_args_and_kw(*args, **kw)))

def pretty_truncate (string, cutoff=100, tail='...') :
    ''' This limits the length of a string in a somewhat aesthetically pleasing fashion. '''

    try :
        return string[:cutoff].rstrip() + tail if len(string) > cutoff else string
    except TypeError :
        # This happens if <cutoff> is None
        return string

_PrettyFloat = type('_PrettyFloat', (float,), {'__repr__' : lambda self : '%0.4f' % self})

def pretty_float (value) :
    ''' Aesthetically pleasing floating point numbers. '''

    return _PrettyFloat(value)

_PrettyEllipsis = type('_PrettyEllipsis', (), {'__repr__' : lambda self : '...'})

def pretty_ellipsis () :
    return _PrettyEllipsis()

def __test__ (ut) :
    ut.assert_equal(pretty_truncate('hello world', 6, '...'), 'hello...')
    ut.assert_equal(pretty_truncate('hello world', 10000, '...'), 'hello world')

    ut.assert_equal(repr(pretty_float(1.30)), '1.3000')

    object, args, kw = ('', (1, 2, 3), {'foo' : 35, 'bar' : 40})

    ut.assert_equal(represented_nonliterally(object, *args, **kw), '<str 1 2 3 bar=40 foo=35>')
    ut.assert_equal(represented_nonliterally(object, *args), '<str 1 2 3>')
    ut.assert_equal(represented_nonliterally(object, **kw), '<str bar=40 foo=35>')
    ut.assert_equal(represented_nonliterally(object), '<str>')

    ut.assert_equal(represented_literally(object, *args, **kw), 'str(1, 2, 3, bar=40, foo=35)')
    ut.assert_equal(represented_literally(object, *args), 'str(1, 2, 3)')
    ut.assert_equal(represented_literally(object, **kw), 'str(bar=40, foo=35)')
    ut.assert_equal(represented_literally(object), 'str()')

if __name__ == '__main__' :
    from nlplib.general.unittest import UnitTest
    __test__(UnitTest())

