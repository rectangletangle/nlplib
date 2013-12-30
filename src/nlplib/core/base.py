

from nlplib.general.represent import represented_nonliterally

__all__ = ['Base']

class Base :
    ''' A base class for all of the core classes. '''

    def __repr__ (self, *args, **kw) :
        # This method can take arguments so subclasses can use the same logic for building their own specific
        # representations.

        return represented_nonliterally(self, *args, **kw)

