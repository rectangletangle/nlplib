

from time import time

from nlplib.general.unit_test import _logging_function

__all__ = ['timing']

def timing (function, log=print) :
    ''' A simple decorator which prints how long a function took to return. '''

    log = _logging_function(log)

    def wrap (*args, **kw) :
        time_0 = time()
        ret = function(*args, **kw)
        time_1 = time()
        log('the function <%s> took %0.3f seconds' % (function.__name__, time_1 - time_0))
        return ret

    return wrap

def __demo__ () :
    from time import sleep

    @timing
    def foo () :
        sleep(0.35)

    foo()

if __name__ == '__main__' :
    __demo__()

