

from nlplib.general import nonliteral_representation

__all__ = ['Base']

class Base :
    def __repr__ (self, *args, **kw) :
        # This method can take arguments so subclasses can use the same logic for building their own specific
        # representations.

        return nonliteral_representation(self, *args, **kw)

