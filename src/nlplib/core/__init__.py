

from itertools import chain

__all__ = ['Base']

class Base :
    def __repr__ (self, *args, **kw) :
        # This method can take arguments so subclasses can use the same logic for building their own specific
        # representations.

        representation = chain((self.__class__.__name__,),
                               (repr(arg) for arg in args),
                               (str(name) + '=' + repr(value) for name, value in kw.items()))

        return '<' + ' '.join(representation) + '>'

