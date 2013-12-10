# todo : lemmatize module names like in core


from itertools import chain

__all__ = ['pretty_truncate', 'nonliteral_representation', 'literal_representation']

def pretty_truncate (string, cutoff=100, tail='...') :
    ''' This limits the length of a string in a somewhat aesthetically pleasing fashion. '''

    try :
        return string[:cutoff].rstrip() + tail if len(string) > cutoff else string
    except TypeError :
        # This happens if <cutoff> is None
        return string

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
    from nlplib.general.unit_test import UnitTest
    __test__(UnitTest())

