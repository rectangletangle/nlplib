

__all__ = ['pretty_truncate']

def pretty_truncate (string, cutoff=100, tail='...') :
    ''' This limits the length of a string in a somewhat aesthetically pleasing fashion. '''

    try :
        return string[:cutoff].rstrip() + tail if len(string) > cutoff else string
    except TypeError :
        # This happens if <cutoff> is None
        return string

def __test__ (ut) :
    ut.assert_equal(pretty_truncate('hello world', 6, '...'), 'hello...')
    ut.assert_equal(pretty_truncate('hello world', 10000, '...'), 'hello world')

if __name__ == '__main__' :
    from nlplib.general.unit_test import UnitTest
    __test__(UnitTest())

